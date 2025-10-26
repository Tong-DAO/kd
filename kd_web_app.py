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

# --- Core Configuration ---
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

# --- Core Data Loading and Utility Functions ---

@st.cache_resource(show_spinner="Syncing data files from the cloud...")
def get_hf_file_path(filename_in_repo):
    try:
        full_path_in_repo = f"data/{filename_in_repo}"
        token = st.secrets.get("HUGGING_FACE_HUB_TOKEN")
        return hf_hub_download(repo_id=HF_REPO_ID, filename=full_path_in_repo, repo_type=HF_REPO_TYPE, use_auth_token=token)
    except Exception as e:
        st.error(f"Failed to download file '{filename_in_repo}': {e}"); st.stop()

# >>>>> 【v1.14 核心修正】: 精准控制缓存，防止内存溢出 <<<<<
@st.cache_data(max_entries=10, ttl=3600)
def load_main_raster_data(filename_in_repo):
    """
    Loads the main raster data for the map.
    max_entries=10: Cache at most 10 raster files. This prevents memory overflow when switching elements.
    ttl=3600: Cache entries expire after 1 hour.
    """
    local_file_path = get_hf_file_path(filename_in_repo)
    try:
        with rasterio.open(local_file_path) as src:
            data = src.read(1).astype(np.float32)
            transform, crs, bounds = src.transform, src.crs, src.bounds
            data[~np.isfinite(data)] = np.nan
            data = np.ma.masked_invalid(data)
            return {'data': data, 'transform': transform, 'crs': crs, 'bounds': bounds}
    except Exception as e:
        st.error(f"Error loading main raster file '{local_file_path}': {e}"); return None

def get_value_from_raster(filename_in_repo, row, col):
    try:
        local_path = get_hf_file_path(filename_in_repo)
        with rasterio.open(local_path) as src:
            return src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0]
    except:
        return None

def wgs84_to_albers(lon, lat, crs):
    try:
        x, y = coord_transform('EPSG:4326', crs, [lon], [lat]); return x[0], y[0]
    except: return None, None

def create_enhanced_colormap():
    colors = ['#00008B', '#0000FF', '#0080FF', '#00BFFF', '#00FF80', '#80FF00', '#FFFF00', '#FF8000', '#FF0000', '#8B0000']
    return LinearSegmentedColormap.from_list('enhanced_viridis', colors, N=256)

def normalize_data(data, method):
    data_copy = np.array(data, copy=True);
    if np.ma.is_masked(data): valid_data = data.compressed()
    else: valid_data = data_copy[np.isfinite(data_copy)]
    if len(valid_data) == 0: return data_copy, 0, 1
    if method == "Raw Data":
        data_copy[data_copy < 0] = 0; return data_copy, 0, np.max(valid_data)
    elif method == "Percentile Normalization":
        p5, p95 = np.percentile(valid_data, 5), np.percentile(valid_data, 95)
        if p95 - p5 > 1e-10: return np.clip((data_copy - p5) / (p95 - p5), 0, 1), 0, 1
    return data_copy, np.min(valid_data), np.max(valid_data)

def create_map_image(display_data, vmin, vmax, element, depth, norm_method, data_info, marker_point=None):
    try:
        fig = plt.figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111); cmap = create_enhanced_colormap() if norm_method != "Raw Data" else 'viridis'
        bounds = data_info['bounds']; width, height = bounds.right - bounds.left, bounds.top - bounds.bottom
        margin_x, margin_y = width * 0.05, height * 0.05
        extent = [bounds.left - margin_x, bounds.right + margin_x, bounds.bottom - margin_y, bounds.top + margin_y]
        im = ax.imshow(display_data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect='auto', interpolation='nearest', origin='upper')
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Kd Value [L/g]', fontsize=10); ax.set_title(f'{element} Kd Distribution in {depth} Soil ({norm_method})', fontsize=12)
        ax.set_xlabel('East Coordinate (m)'); ax.set_ylabel('North Coordinate (m)'); ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)
        if marker_point:
            row, col = marker_point; x, y = rasterio.transform.xy(data_info['transform'], row, col)
            ax.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
            ax.annotate('Query Point', xy=(x, y), xytext=(x + width * 0.02, y - height * 0.02), fontsize=9, color='red', arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        plt.tight_layout(pad=2.0); buf = BytesIO(); plt.savefig(buf, format='png', dpi=100, bbox_inches='tight'); buf.seek(0); plt.close(fig)
        return buf
    except Exception as e:
        st.error(f"Error creating map image: {e}"); return None

@st.cache_data
def get_all_point_data(lon, lat, element, crs_str, transform_tuple, shape_tuple):
    try:
        crs = rasterio.crs.CRS.from_string(crs_str)
        transform = rasterio.Affine.from_gdal(*transform_tuple)
        shape = shape_tuple
        x, y = wgs84_to_albers(lon, lat, crs)
        if x is None: return None, None
        row, col = rasterio.transform.rowcol(transform, x, y)
        if not (0 <= row < shape[0] and 0 <= col < shape[1]): return None, None

        depths = {"0-5cm": "05", "5-15cm": "515", "15-30cm": "1530", "30-60cm": "3060", "60-100cm": "60100"}
        all_data = {}
        for depth_label, depth_suffix in depths.items():
            params = {}
            kd_file = f"prediction_result_{element}{depth_suffix}_raw.tif"
            kd_value = get_value_from_raster(kd_file, int(row), int(col))
            if kd_value is None or not np.isfinite(kd_value): continue
            params['Kd'] = float(kd_value)
            
            # ... (rest of the parameter fetching logic is fine)
            param_files = {"pH": f"ph{depth_suffix}.tif", "SOM": f"soc{depth_suffix}.tif", "CEC": f"cec{depth_suffix}.tif"}
            for param_name, filename in param_files.items():
                value = get_value_from_raster(filename, int(row), int(col))
                if value is not None and np.isfinite(value):
                    if param_name in ["pH", "CEC"]: value /= 100
                    elif param_name == "SOM": value = value * 1.724 / 100
                    params[param_name] = float(value)
            ec_file = "T_ECE.tif" if depth_suffix in ["05", "515", "1530"] else "S_ECE.tif"
            ec_value = get_value_from_raster(ec_file, int(row), int(col))
            if ec_value is not None and np.isfinite(ec_value): params["IS"] = float(max(0.0446 * ec_value - 0.000173, 0))
            ce_file = f"{element}.tif"
            ce_value = get_value_from_raster(ce_file, int(row), int(col))
            if ce_value is not None and np.isfinite(ce_value): params["Ce"] = float(ce_value)
            
            all_data[depth_label] = params
        return all_data, (int(row), int(col))
    except:
        return None, None

def create_depth_profile_chart(depth_profile_data, element):
    if not depth_profile_data: return None
    df = pd.DataFrame(depth_profile_data).T.reset_index()
    df.rename(columns={'index': 'Depth'}, inplace=True)
    if 'Kd' not in df.columns: return None
    
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(df['Depth'], df['Kd'], marker='o', linestyle='-'); ax.set_xlabel('Soil Depth'); ax.set_ylabel('Kd Value [L/g]')
    ax.set_title(f'Kd Value Depth Profile for {element}'); ax.grid(True, alpha=0.3); plt.tight_layout()
    return fig

# --- UI Sidebar ---
with st.sidebar:
    st.header("📊 Parameter Settings")
    element = st.selectbox("Rare Earth Element", ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y"], help="Select a rare earth element to display")
    depth = st.selectbox("Soil Depth", ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"], help="Select a soil sampling depth")
    norm_method = st.selectbox("Normalization Method", ["Raw Data", "Percentile Normalization", "Standard Deviation Normalization", "Linear Normalization"], help="Select a data normalization method")
    st.markdown("---"); st.header("🔍 Coordinate Query")
    lon = st.number_input("Longitude", min_value=73.0, max_value=135.0, value=105.0, step=0.1)
    lat = st.number_input("Latitude", min_value=18.0, max_value=53.0, value=35.0, step=0.1)
    query_button = st.button("🎯 Query Point", use_container_width=True, type="primary")
    st.markdown("---"); show_stats = st.checkbox("Show Statistics", value=False)

# --- Main Logic ---
depth_mapping = {"0-5cm": "05", "5-15cm": "515", "15-30cm": "1530", "30-60cm": "3060", "60-100cm": "60100"}
depth_suffix = depth_mapping[depth]; raster_filename = f"prediction_result_{element}{depth_suffix}_raw.tif"
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Kd Value Spatial Distribution")
    main_data_info = load_main_raster_data(raster_filename)
    if main_data_info is None: st.error("Could not load main data file. Please refresh the page."); st.stop()
    
    if show_stats:
        valid_data = main_data_info['data'].compressed() if np.ma.is_masked(main_data_info['data']) else main_data_info['data'][np.isfinite(main_data_info['data'])]
        if len(valid_data) > 0:
            stats_cols = st.columns(4); stats_cols[0].metric("Min", f"{np.min(valid_data):.4f}"); stats_cols[1].metric("Max", f"{np.max(valid_data):.4f}"); stats_cols[2].metric("Mean", f"{np.mean(valid_data):.4f}"); stats_cols[3].metric("Median", f"{np.median(valid_data):.4f}")
    
    if query_button:
        st.session_state['query_result'] = None; st.session_state['depth_profile_data'] = None; st.session_state['marker_point'] = None
        with st.spinner("Querying all parameters for the selected point..."):
            crs_str = main_data_info['crs'].to_string()
            transform_tuple = main_data_info['transform'].to_gdal()
            shape_tuple = main_data_info['data'].shape
            all_data, marker = get_all_point_data(lon, lat, element, crs_str, transform_tuple, shape_tuple)
        if all_data:
            st.session_state['query_result'] = all_data.get(depth)
            st.session_state['depth_profile_data'] = all_data
            st.session_state['marker_point'] = marker

    marker_point_to_display = st.session_state.get('marker_point', None)
    
    with st.spinner('Generating map...'):
        display_data, vmin, vmax = normalize_data(main_data_info['data'], norm_method)
        img_buf = create_map_image(display_data, vmin, vmax, element, depth, norm_method, main_data_info, marker_point_to_display)
    if img_buf: st.image(img_buf, use_container_width=True)
    else: st.error("Failed to generate map image.")

with col_right:
    tab1, tab2 = st.tabs(["📍 Query Results", "📈 Depth Profile Analysis"])
    with tab1:
        # ... (This part of the UI remains the same and is stable)
        if 'query_result' in st.session_state and st.session_state.get('query_result'):
            params = st.session_state['query_result']; st.success("✅ Query Successful")
            st.markdown(f"**📍 Location Information**\n- Longitude: {lon:.4f}°E\n- Latitude: {lat:.4f}°N\n- Element: {element}\n- Depth: {depth}")
            st.markdown("---"); st.markdown("**📊 Soil Parameters**")
            param_display_df = pd.DataFrame(params.items(), columns=["Parameter", "Value"])
            param_display_df['Value'] = param_display_df['Value'].apply(lambda v: f"{v:.2f}" if v >= 1 else f"{v:.4f}")
            st.dataframe(param_display_df, hide_index=True, use_container_width=True)
            csv = param_display_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results as CSV", csv, f'query_results_{element}_{lon}_{lat}_{depth}.csv', 'text/csv', use_container_width=True)
        else:
            if query_button: st.warning("⚠️ No valid data at this location or for the selected depth.")
            else: st.info("👆 Enter coordinates and click 'Query Point'.")
            st.markdown("**📊 Soil Parameters**"); empty_df = pd.DataFrame({"Parameter": ["Kd", "pH", "SOM", "CEC", "IS", "Ce"], "Value": ["--"] * 6})
            st.dataframe(empty_df, hide_index=True, use_container_width=True)

    with tab2:
        if 'depth_profile_data' in st.session_state and st.session_state.get('depth_profile_data'):
            profile_data = st.session_state['depth_profile_data']
            st.info("Depth profile data is available based on your latest query.")
            profile_chart = create_depth_profile_chart(profile_data, element)
            if profile_chart: st.pyplot(profile_chart)
            else: st.warning("Could not create profile chart. Not enough data points across depths.")
        else:
            st.warning("Please perform a successful query to view depth profile data.")

# --- v1.14 Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>🌱 REEs Soil Kd Visualization System v1.14<br>Final Memory-Managed Stable Version</div>", unsafe_allow_html=True)