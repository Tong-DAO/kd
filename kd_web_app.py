import streamlit as st
import numpy as np
import rasterio
from rasterio.warp import transform as coord_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from io import BytesIO
import warnings
from huggingface_hub import hf_hub_download

warnings.filterwarnings('ignore')

# --- v1.0 (Core Configuration) ---
# This block contains the stable, core configuration of the app.
HF_REPO_ID = "Tong-DAO/REES"
HF_REPO_TYPE = "dataset"
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="REEs Soil Kd Value Visualization",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌱 REEs Soil Kd Value Visualization System")

# --- v1.0 (Core Data Loading Functions) ---
# These functions are the stable foundation for data access.
@st.cache_resource(show_spinner="Syncing data files from the cloud...")
def get_hf_file_path(filename_in_repo):
    try:
        full_path_in_repo = f"data/{filename_in_repo}"
        token = st.secrets.get("HUGGING_FACE_HUB_TOKEN")
        return hf_hub_download(repo_id=HF_REPO_ID, filename=full_path_in_repo, repo_type=HF_REPO_TYPE, use_auth_token=token)
    except Exception as e:
        st.error(f"Failed to download file '{filename_in_repo}': {e}")
        st.stop()

@st.cache_data(show_spinner="Loading and parsing raster data...")
def load_raster_data(filename_in_repo):
    local_file_path = get_hf_file_path(filename_in_repo)
    try:
        with rasterio.open(local_file_path) as src:
            data = src.read(1).astype(np.float32)
            transform_matrix, crs, bounds = src.transform, src.crs, src.bounds
            data[~np.isfinite(data)] = np.nan
            data = np.ma.masked_invalid(data)
            return {'data': data, 'transform': transform_matrix, 'crs': crs, 'bounds': bounds}
    except Exception as e:
        st.error(f"Error loading raster file '{local_file_path}': {e}")
        return None

# --- v1.0 (Core Utility and Processing Functions) ---
# These helper functions perform stable, reusable tasks.
def wgs84_to_albers(lon, lat, crs):
    try:
        x, y = coord_transform('EPSG:4326', crs, [lon], [lat])
        return x[0], y[0]
    except: return None, None

def create_enhanced_colormap():
    colors = ['#00008B', '#0000FF', '#0080FF', '#00BFFF', '#00FF80', '#80FF00', '#FFFF00', '#FF8000', '#FF0000', '#8B0000']
    return LinearSegmentedColormap.from_list('enhanced_viridis', colors, N=256)

def normalize_data(data, method):
    data_copy = np.array(data, copy=True)
    if np.ma.is_masked(data): valid_data = data.compressed()
    else: valid_data = data_copy[np.isfinite(data_copy)]
    if len(valid_data) == 0: return data_copy, 0, 1
    if method == "Raw Data":
        data_copy[data_copy < 0] = 0; return data_copy, 0, np.max(valid_data)
    elif method == "Percentile Normalization":
        p5, p95 = np.percentile(valid_data, 5), np.percentile(valid_data, 95)
        if p95 - p5 > 1e-10: return np.clip((data_copy - p5) / (p95 - p5), 0, 1), 0, 1
    elif method == "Standard Deviation Normalization":
        mean, std = np.mean(valid_data), np.std(valid_data)
        if std > 1e-10: return np.clip((data_copy - mean) / (2 * std) + 0.5, 0, 1), 0, 1
    elif method == "Linear Normalization":
        min_val, max_val = np.min(valid_data), np.max(valid_data)
        if max_val - min_val > 1e-10: return (data_copy - min_val) / (max_val - min_val), 0, 1
    return data_copy, np.min(valid_data), np.max(valid_data)

def get_point_parameters(lon, lat, element, depth_suffix, data_info_for_crs):
    try:
        x, y = wgs84_to_albers(lon, lat, data_info_for_crs['crs'])
        if x is None: return None
        row, col = rasterio.transform.rowcol(data_info_for_crs['transform'], x, y)
        if not (0 <= row < data_info_for_crs['data'].shape[0] and 0 <= col < data_info_for_crs['data'].shape[1]): return None
        
        main_raster_file = f"prediction_result_{element}{depth_suffix}_raw.tif"
        main_data_info = load_raster_data(main_raster_file)
        kd_value = main_data_info['data'][row, col]
        if np.ma.is_masked(kd_value) or not np.isfinite(kd_value): return None

        params = {"Kd": float(kd_value)}
        param_files = {"pH": f"ph{depth_suffix}.tif", "SOM": f"soc{depth_suffix}.tif", "CEC": f"cec{depth_suffix}.tif", "Ce": f"{element}.tif"}
        for param_name, filename in param_files.items():
            try:
                param_data = load_raster_data(filename)
                value = param_data['data'][row, col]
                if param_name in ["pH", "CEC"]: value /= 100
                elif param_name == "SOM": value = value * 1.724 / 100
                params[param_name] = float(value)
            except: st.toast(f"Failed to load auxiliary parameter: {param_name}", icon="⚠️")
        
        ec_file = "T_ECE.tif" if depth_suffix in ["05", "515", "1530"] else "S_ECE.tif"
        try:
            ec_data = load_raster_data(ec_file)
            ec_value = ec_data['data'][row, col]
            params["IS"] = float(max(0.0446 * ec_value - 0.000173, 0))
        except: st.toast("Failed to calculate Ionic Strength (IS)", icon="⚠️")
        return params, (row, col)
    except: return None

def create_map_image(display_data, vmin, vmax, element, depth, norm_method, data_info, marker_point=None):
    try:
        fig = plt.figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)
        cmap = create_enhanced_colormap() if norm_method != "Raw Data" else 'viridis'
        bounds = data_info['bounds']
        width, height = bounds.right - bounds.left, bounds.top - bounds.bottom
        margin_x, margin_y = width * 0.05, height * 0.05
        extent = [bounds.left - margin_x, bounds.right + margin_x, bounds.bottom - margin_y, bounds.top + margin_y]
        im = ax.imshow(display_data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect='auto', interpolation='nearest', origin='upper')
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Kd Value [L/g]', fontsize=10)
        ax.set_title(f'{element} Kd Distribution in {depth} Soil ({norm_method})', fontsize=12)
        ax.set_xlabel('East Coordinate (m)'); ax.set_ylabel('North Coordinate (m)')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)
        if marker_point:
            row, col = marker_point
            x, y = rasterio.transform.xy(data_info['transform'], row, col)
            ax.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
            ax.annotate('Query Point', xy=(x, y), xytext=(x + width * 0.02, y - height * 0.02), fontsize=9, color='red', arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        plt.tight_layout(pad=2.0)
        buf = BytesIO(); plt.savefig(buf, format='png', dpi=100, bbox_inches='tight'); buf.seek(0); plt.close(fig)
        return buf
    except Exception as e:
        st.error(f"Error creating map image: {e}"); return None

# --- v1.02 NEW FEATURES ---
# This block contains the new, modular functions for v1.02.
@st.cache_data(show_spinner="Generating depth profile...")
def get_depth_profile_data(lon, lat, element, base_data_info):
    """Fetches Kd values for all depths at a single point."""
    depths = {"0-5cm": "05", "5-15cm": "515", "15-30cm": "1530", "30-60cm": "3060", "60-100cm": "60100"}
    profile_data = {}
    try:
        x, y = wgs84_to_albers(lon, lat, base_data_info['crs'])
        if x is None: return None
        row, col = rasterio.transform.rowcol(base_data_info['transform'], x, y)
        if not (0 <= row < base_data_info['data'].shape[0] and 0 <= col < base_data_info['data'].shape[1]): return None

        for depth_label, depth_suffix in depths.items():
            raster_file = f"prediction_result_{element}{depth_suffix}_raw.tif"
            data_info = load_raster_data(raster_file)
            kd_value = data_info['data'][row, col]
            if not (np.ma.is_masked(kd_value) or not np.isfinite(kd_value)):
                profile_data[depth_label] = float(kd_value)
        return profile_data
    except:
        return None

def create_depth_profile_chart(profile_data, element):
    """Creates a line chart for the depth profile."""
    if not profile_data: return None
    df = pd.DataFrame(list(profile_data.items()), columns=['Depth', 'Kd Value'])
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(df['Depth'], df['Kd Value'], marker='o', linestyle='-')
    ax.set_xlabel('Soil Depth')
    ax.set_ylabel('Kd Value [L/g]')
    ax.set_title(f'Kd Value Depth Profile for {element}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# --- v1.0 (Core UI Sidebar) ---
# The sidebar is part of the stable core UI.
with st.sidebar:
    st.header("📊 Parameter Settings")
    element = st.selectbox("Rare Earth Element", ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y"], key="element_select")
    depth = st.selectbox("Soil Depth", ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"], key="depth_select")
    norm_method = st.selectbox("Normalization Method", ["Raw Data", "Percentile Normalization", "Standard Deviation Normalization", "Linear Normalization"], key="norm_select")
    st.markdown("---")
    st.header("🔍 Coordinate Query")
    lon = st.number_input("Longitude", min_value=73.0, max_value=135.0, value=105.0, step=0.1, key="lon_input")
    lat = st.number_input("Latitude", min_value=18.0, max_value=53.0, value=35.0, step=0.1, key="lat_input")
    query_button = st.button("🎯 Query Point", use_container_width=True, type="primary")
    st.markdown("---")
    show_stats = st.checkbox("Show Statistics", value=False)

# --- v1.0 (Core Main Logic) ---
# This is the main part of the app that orchestrates everything.
depth_mapping = {"0-5cm": "05", "5-15cm": "515", "15-30cm": "1530", "30-60cm": "3060", "60-100cm": "60100"}
depth_suffix = depth_mapping[depth]
raster_filename = f"prediction_result_{element}{depth_suffix}_raw.tif"

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Kd Value Spatial Distribution")
    data_info = load_raster_data(raster_filename)
    if data_info is None:
        st.error("Could not load data file. Please refresh the page to try again.")
        st.stop()
    
    if show_stats:
        valid_data = data_info['data'].compressed() if np.ma.is_masked(data_info['data']) else data_info['data'][np.isfinite(data_info['data'])]
        if len(valid_data) > 0:
            stats_cols = st.columns(4); stats_cols[0].metric("Min", f"{np.min(valid_data):.4f}"); stats_cols[1].metric("Max", f"{np.max(valid_data):.4f}"); stats_cols[2].metric("Mean", f"{np.mean(valid_data):.4f}"); stats_cols[3].metric("Median", f"{np.median(valid_data):.4f}")
    
    if query_button:
        result = get_point_parameters(lon, lat, element, depth_suffix, data_info)
        st.session_state['query_result'], st.session_state['marker_point'] = result if result else (None, None)
    
    marker_point_to_display = st.session_state.get('marker_point', None)
    
    with st.spinner('Generating map...'):
        display_data, vmin, vmax = normalize_data(data_info['data'], norm_method)
        img_buf = create_map_image(display_data, vmin, vmax, element, depth, norm_method, data_info, marker_point_to_display)
    if img_buf: st.image(img_buf, use_container_width=True)
    else: st.error("Failed to generate map image.")

with col_right:
    # --- v1.02 UI REFACTOR: Using Tabs ---
    tab1, tab2 = st.tabs(["📍 Query Results", "📈 Depth Profile Analysis"])

    with tab1:
        # This part is mostly the original v1.0 query result display.
        if 'query_result' in st.session_state and st.session_state['query_result'] is not None:
            params = st.session_state['query_result']
            st.success("✅ Query Successful")
            st.markdown(f"**📍 Location Information**\n- Longitude: {lon:.4f}°E\n- Latitude: {lat:.4f}°N\n- Element: {element}\n- Depth: {depth}")
            st.markdown("---")
            st.markdown("**📊 Soil Parameters**")
            
            param_display_df = pd.DataFrame([{"Parameter": k, "Value": f"{v:.2f}" if v >= 1 else f"{v:.4f}"} for k, v in params.items()])
            st.dataframe(param_display_df, hide_index=True, use_container_width=True)

            # --- v1.02 NEW FEATURE: Download Button ---
            csv = param_display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f'query_results_{element}_{lon}_{lat}.csv',
                mime='text/csv',
                use_container_width=True
            )
            # --- End of Download Button Feature ---
            
            with st.expander("📖 Parameter Description"):
                st.markdown("- **Kd**: Distribution coefficient\n- **pH**: Soil acidity/alkalinity\n- **SOM**: Soil organic matter\n- **CEC**: Cation exchange capacity\n- **IS**: Ionic strength\n- **Ce**: Equilibrium concentration")
        else:
            if query_button: st.warning("⚠️ No valid data at this location or out of range.")
            else: st.info("👆 Enter coordinates and click 'Query Point'.")
            st.markdown("**📊 Soil Parameters**")
            empty_df = pd.DataFrame({"Parameter": ["Kd", "pH", "SOM", "CEC", "IS", "Ce"], "Value": ["--"] * 6})
            st.dataframe(empty_df, hide_index=True, use_container_width=True)

    with tab2:
        # --- v1.02 NEW FEATURE: Depth Profile Section ---
        st.header("Vertical Kd Distribution")
        st.info("Click the button below to analyze the Kd value variation with depth at the queried location.")
        
        # Only show the button if a point has been successfully queried
        if 'query_result' in st.session_state and st.session_state['query_result'] is not None:
            if st.button("Generate Depth Profile", use_container_width=True, type="primary"):
                profile_data = get_depth_profile_data(lon, lat, element, data_info)
                if profile_data:
                    profile_chart = create_depth_profile_chart(profile_data, element)
                    st.pyplot(profile_chart)
                else:
                    st.error("Could not generate profile. Data might be missing for some depths at this location.")
        else:
            st.warning("Please query a point first in the 'Query Results' tab.")
        # --- End of Depth Profile Feature ---

# --- v1.02 NEW FEATURE: Updated Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>🌱 REEs Soil Kd Visualization System v1.02<br>Now with Depth Profile Analysis and Data Download</div>", unsafe_allow_html=True)