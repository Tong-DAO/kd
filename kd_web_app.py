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

# --- v3.0: Core Configuration & Functions (Based on your original stable code) ---
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

# --- v3.0: Cloud-Adapted Data Loading ---
@st.cache_data(show_spinner="Syncing data from cloud...")
def get_raster_from_hf(filename_in_repo):
    """
    Downloads a raster file from Hugging Face and opens it with rasterio.
    Returns the rasterio dataset object.
    """
    try:
        full_path_in_repo = f"data/{filename_in_repo}"
        token = st.secrets.get("HUGGING_FACE_HUB_TOKEN")
        local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=full_path_in_repo, repo_type=HF_REPO_TYPE, use_auth_token=token)
        return rasterio.open(local_path)
    except Exception as e:
        st.warning(f"Could not load data for '{filename_in_repo}': {e}", icon="⚠️")
        return None

# --- v3.0: Utility Functions from your original code ---
def wgs84_to_albers(lon, lat, crs):
    try:
        x, y = coord_transform('EPSG:4326', crs, [lon], [lat]); return x[0], y[0]
    except: return None, None

def create_enhanced_colormap():
    colors = ['#00008B', '#0000FF', '#0080FF', '#00BFFF', '#00FF80', '#80FF00', '#FFFF00', '#FF8000', '#FF0000', '#8B0000']
    return LinearSegmentedColormap.from_list('enhanced_viridis', colors, N=256)

def normalize_data(data, method):
    data_copy = np.array(data, copy=True)
    valid_data = data.compressed() if np.ma.is_masked(data) else data_copy[np.isfinite(data_copy)]
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
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02); cbar.set_label('Kd Value [L/g]', fontsize=10)
        ax.set_title(f'{element} Kd Distribution in {depth} Soil ({norm_method})', fontsize=12)
        ax.set_xlabel('East Coordinate (m)'); ax.set_ylabel('North Coordinate (m)'); ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)
        if marker_point:
            row, col = marker_point; x, y = rasterio.transform.xy(data_info['transform'], row, col)
            ax.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
            ax.annotate('Query Point', xy=(x, y), xytext=(x + width * 0.02, y - height * 0.02), fontsize=9, color='red', arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        plt.tight_layout(pad=2.0); buf = BytesIO(); plt.savefig(buf, format='png', dpi=100, bbox_inches='tight'); buf.seek(0); plt.close(fig)
        return buf
    except Exception as e:
        st.error(f"Error creating map image: {e}"); return None

# --- v3.0: UI Sidebar (Original simple structure) ---
with st.sidebar:
    st.header("📊 Parameter Settings")
    element = st.selectbox("Rare Earth Element", ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y"], help="Select rare earth element to display")
    depth = st.selectbox("Soil Depth", ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"], help="Select soil sampling depth")
    norm_method = st.selectbox("Normalization Method", ["Raw Data", "Percentile Normalization", "Standard Deviation Normalization", "Linear Normalization"], help="Select data normalization method")
    st.markdown("---"); st.header("🔍 Coordinate Query")
    lon = st.number_input("Longitude", min_value=73.0, max_value=135.0, value=105.0, step=0.1)
    lat = st.number_input("Latitude", min_value=18.0, max_value=53.0, value=35.0, step=0.1)
    query_button = st.button("🎯 Query Point", use_container_width=True, type="primary")
    st.markdown("---"); show_stats = st.checkbox("Show Statistics", value=False)

# --- v3.0: Main Logic (Reflecting your original simple script flow) ---
depth_mapping = {"0-5cm": "05", "5-15cm": "515", "15-30cm": "1530", "30-60cm": "3060", "60-100cm": "60100"}
depth_suffix = depth_mapping[depth]; raster_filename = f"prediction_result_{element}{depth_suffix}_raw.tif"

col_left, col_right = st.columns([2, 1])

# --- Main Map Display ---
with col_left:
    st.subheader("📊 Kd Value Spatial Distribution")
    main_src = get_raster_from_hf(raster_filename)
    if main_src is None: st.error(f"❌ Main data file not found or failed to load: {raster_filename}"); st.stop()
    
    with main_src: # Use 'with' statement for proper resource management
        main_data = main_src.read(1).astype(np.float32)
        main_data[~np.isfinite(main_data)] = np.nan; main_data = np.ma.masked_invalid(main_data)
        data_info = {'data': main_data, 'transform': main_src.transform, 'crs': main_src.crs, 'bounds': main_src.bounds}

    if show_stats:
        valid_data = data_info['data'].compressed()
        if len(valid_data) > 0:
            stats_cols = st.columns(4); stats_cols[0].metric("Min", f"{np.min(valid_data):.4f}"); stats_cols[1].metric("Max", f"{np.max(valid_data):.4f}"); stats_cols[2].metric("Mean", f"{np.mean(valid_data):.4f}"); stats_cols[3].metric("Median", f"{np.median(valid_data):.4f}")
    
    display_data, vmin, vmax = normalize_data(data_info['data'], norm_method)
    
    # Query logic is now also simplified and self-contained
    marker_point = None
    query_params = None
    if query_button:
        with st.spinner("Querying point..."):
            x, y = wgs84_to_albers(lon, lat, data_info['crs'])
            if x is not None:
                row, col = rasterio.transform.rowcol(data_info['transform'], x, y)
                if (0 <= row < data_info['data'].shape[0] and 0 <= col < data_info['data'].shape[1]):
                    kd_value = data_info['data'][row, col]
                    if not (np.ma.is_masked(kd_value) or not np.isfinite(kd_value)):
                        marker_point = (row, col)
                        query_params = {"Kd": float(kd_value)}
                        # Load other params on the fly
                        param_files = {"pH": f"ph{depth_suffix}.tif", "SOM": f"soc{depth_suffix}.tif", "CEC": f"cec{depth_suffix}.tif", "Ce": f"{element}.tif"}
                        for p_name, fname in param_files.items():
                            with get_raster_from_hf(fname) as p_src:
                                if p_src:
                                    value = p_src.read(1)[row, col]
                                    if p_name in ["pH", "CEC"]: value /= 100
                                    elif p_name == "SOM": value = value * 1.724 / 100
                                    query_params[p_name] = float(value)
                        ec_file = "T_ECE.tif" if depth_suffix in ["05", "515", "1530"] else "S_ECE.tif"
                        with get_raster_from_hf(ec_file) as ec_src:
                            if ec_src:
                                ec_value = ec_src.read(1)[row, col]
                                query_params["IS"] = float(max(0.0446 * ec_value - 0.000173, 0))

    with st.spinner('Generating map...'):
        img_buf = create_map_image(display_data, vmin, vmax, element, depth, norm_method, data_info, marker_point)
    if img_buf: st.image(img_buf, use_container_width=True)
    else: st.error("Failed to generate map image.")

# --- Query Results Display ---
with col_right:
    st.subheader("📍 Query Results")
    if query_params:
        st.success("✅ Query Successful")
        st.markdown(f"**📍 Location Information**\n- Longitude: {lon:.4f}°E\n- Latitude: {lat:.4f}°N\n- Element: {element}\n- Depth: {depth}")
        st.markdown("---"); st.markdown("**📊 Soil Parameters**")
        param_display_df = pd.DataFrame(query_params.items(), columns=["Parameter", "Value"])
        param_display_df['Value'] = param_display_df['Value'].apply(lambda v: f"{v:.2f}" if v >= 1 else f"{v:.4f}")
        st.dataframe(param_display_df, hide_index=True, use_container_width=True)
    else:
        if query_button: st.warning("⚠️ No valid data at this location or query failed.")
        else: st.info("👆 Enter coordinates and click 'Query Point'.")
        st.markdown("**📊 Soil Parameters**"); empty_df = pd.DataFrame({"Parameter": ["Kd", "pH", "SOM", "CEC", "IS", "Ce"], "Value": ["--"] * 6})
        st.dataframe(empty_df, hide_index=True, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>🌱 REEs Soil Kd Visualization System v3.0<br>Baseline Stable Version</div>", unsafe_allow_html=True)