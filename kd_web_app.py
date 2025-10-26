import streamlit as st
import numpy as np
import rasterio
from rasterio.warp import transform as coord_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from io import BytesIO
import warnings
from huggingface_hub import hf_hub_download

warnings.filterwarnings('ignore')

# --- Core Configuration & Functions (The "Keep") ---
HF_REPO_ID = "Tong-DAO/REES"
HF_REPO_TYPE = "dataset"
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="REEs Soil Kd Value Visualization", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")
st.title("🌱 REEs Soil Kd Value Visualization System")

@st.cache_resource(show_spinner="Syncing data files from the cloud...")
def get_hf_file_path(filename_in_repo):
    try:
        full_path_in_repo = f"data/{filename_in_repo}"; token = st.secrets.get("HUGGING_FACE_HUB_TOKEN")
        return hf_hub_download(repo_id=HF_REPO_ID, filename=full_path_in_repo, repo_type=HF_REPO_TYPE, use_auth_token=token)
    except Exception as e:
        st.error(f"Critical Error: Failed to download file '{filename_in_repo}': {e}"); st.stop()

@st.cache_data(max_entries=5) # 进一步减少缓存，确保内存在最安全范围
def load_main_raster_data(filename_in_repo):
    local_file_path = get_hf_file_path(filename_in_repo)
    try:
        with rasterio.open(local_file_path) as src:
            data = src.read(1).astype(np.float32); transform, crs, bounds = src.transform, src.crs, src.bounds
            data[~np.isfinite(data)] = np.nan; data = np.ma.masked_invalid(data)
            return {'data': data, 'transform': transform, 'crs': crs, 'bounds': bounds}
    except Exception as e:
        st.error(f"Error loading main raster file '{local_file_path}': {e}"); return None

def get_value_from_raster(filename_in_repo, row, col):
    try:
        local_path = get_hf_file_path(filename_in_repo)
        with rasterio.open(local_path) as src:
            return src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0]
    except: return None

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

# --- Fallback (Safe) Version of Map Creation ---
def create_map_fallback(display_data, vmin, vmax, data_info, title_text):
    """This is the absolutely safe, original map creation function."""
    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax = fig.add_subplot(111); cmap = create_enhanced_colormap()
    extent = [data_info['bounds'].left, data_info['bounds'].right, data_info['bounds'].bottom, data_info['bounds'].top]
    im = ax.imshow(display_data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect='auto', origin='upper')
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Kd Value [L/g]'); ax.set_title(title_text)
    ax.set_xlabel('East Coordinate (m)'); ax.set_ylabel('North Coordinate (m)')
    plt.tight_layout(); buf = BytesIO(); plt.savefig(buf, format='png'); buf.seek(0); plt.close(fig)
    return buf

# --- Advanced (Experimental) Version of Map Creation with Lat/Lon ---
def create_map_with_latlon(display_data, vmin, vmax, data_info, title_text, marker_point=None):
    """Advanced map with Lat/Lon axes. Wrapped in a try-except block."""
    fig = plt.figure(figsize=(12, 10), dpi=150)
    ax = fig.add_subplot(111)
    cmap = create_enhanced_colormap() if st.session_state.norm_method != "Raw Data" else 'viridis'
    extent = [data_info['bounds'].left, data_info['bounds'].right, data_info['bounds'].bottom, data_info['bounds'].top]
    im = ax.imshow(display_data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect='auto', origin='upper')
    
    lon_ticks_wgs = np.arange(70, 140, 10); lat_ticks_wgs = np.arange(15, 60, 5)
    x_ticks_albers, _ = wgs84_to_albers(lon_ticks_wgs, [data_info['crs'].standard_parallels[0]]*len(lon_ticks_wgs), data_info['crs'])
    _, y_ticks_albers = wgs84_to_albers([data_info['crs'].central_meridian]*len(lat_ticks_wgs), lat_ticks_wgs, data_info['crs'])
    
    ax.set_xticks(x_ticks_albers); ax.set_yticks(y_ticks_albers)
    ax.set_xticklabels([f"{lon}°E" for lon in lon_ticks_wgs]); ax.set_yticklabels([f"{lat}°N" for lat in lat_ticks_wgs])
    
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel('Longitude', fontsize=12); ax.set_ylabel('Latitude', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_title(title_text, fontsize=14, pad=20); ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    if marker_point:
        row, col = marker_point; x, y = rasterio.transform.xy(data_info['transform'], row, col)
        ax.plot(x, y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        ax.annotate('Query', xy=(x, y), xytext=(x + 50000, y - 50000), color='red', arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8); cbar.set_label('Kd Value [L/g]', fontsize=12)
    plt.tight_layout(); buf = BytesIO(); plt.savefig(buf, format='png'); buf.seek(0); plt.close(fig)
    return buf

# --- Advanced (Experimental) Query Functions ---
@st.cache_data
def get_all_point_data(lon, lat, element, crs_str, transform_tuple, shape_tuple):
    try:
        crs = rasterio.crs.CRS.from_string(crs_str); transform = rasterio.Affine.from_gdal(*transform_tuple)
        x, y = wgs84_to_albers(lon, lat, crs)
        if x is None: return None, None
        row, col = rasterio.transform.rowcol(transform, x, y)
        if not (0 <= row < shape_tuple[0] and 0 <= col < shape_tuple[1]): return None, None

        depths = {"0-5cm": "05", "5-15cm": "515", "15-30cm": "1530", "30-60cm": "3060", "60-100cm": "60100"}
        all_data = {}
        for depth_label, depth_suffix in depths.items():
            params = {}
            kd_file = f"prediction_result_{element}{depth_suffix}_raw.tif"; kd_value = get_value_from_raster(kd_file, int(row), int(col))
            if kd_value is None or not np.isfinite(kd_value): continue
            params['Kd'] = float(kd_value)
            all_data[depth_label] = params
        return all_data, (int(row), int(col))
    except: return None, None

def create_depth_profile_chart(depth_profile_data, element):
    df = pd.DataFrame(depth_profile_data).T.reset_index(); df.rename(columns={'index': 'Depth'}, inplace=True)
    if 'Kd' not in df.columns: return None
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(df['Depth'], df['Kd'], marker='o', linestyle='-'); ax.set_xlabel('Soil Depth'); ax.set_ylabel('Kd Value [L/g]')
    ax.set_title(f'Kd Value Depth Profile for {element}'); ax.grid(True, alpha=0.3); plt.tight_layout()
    return fig

# --- UI Sidebar ---
with st.sidebar:
    st.header("📊 Parameter Settings")
    st.selectbox("Rare Earth Element", ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y"], help="Select a rare earth element", key='element')
    st.selectbox("Soil Depth", ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"], help="Select a soil sampling depth", key='depth')
    st.selectbox("Normalization Method", ["Raw Data", "Percentile Normalization", "Standard Deviation Normalization"], help="Select a data normalization method", key='norm_method')
    st.markdown("---"); st.header("🔍 Coordinate Query")
    st.number_input("Longitude", min_value=73.0, max_value=135.0, value=105.0, step=0.1, key='lon')
    st.number_input("Latitude", min_value=18.0, max_value=53.0, value=35.0, step=0.1, key='lat')
    st.button("🎯 Query Point", use_container_width=True, type="primary", key='query_button')
    st.markdown("---"); st.checkbox("Show Statistics", value=False, key='show_stats')

# --- Main Logic with Fault Tolerance ---
depth_mapping = {"0-5cm": "05", "5-15cm": "515", "15-30cm": "1530", "30-60cm": "3060", "60-100cm": "60100"}
depth_suffix = depth_mapping[st.session_state.depth]; raster_filename = f"prediction_result_{st.session_state.element}{depth_suffix}_raw.tif"
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Kd Value Spatial Distribution")
    main_data_info = load_main_raster_data(raster_filename)
    if main_data_info is None: st.error("Could not load main data file. Please refresh the page."); st.stop()

    if st.session_state.show_stats:
        # ... (statistics logic is safe)
        pass
    
    if st.session_state.query_button:
        # Simplified state management
        with st.spinner("Querying all parameters for the selected point..."):
            crs_str = main_data_info['crs'].to_string(); transform_tuple = main_data_info['transform'].to_gdal(); shape_tuple = main_data_info['data'].shape
            all_data, marker = get_all_point_data(st.session_state.lon, st.session_state.lat, st.session_state.element, crs_str, transform_tuple, shape_tuple)
            st.session_state['all_query_data'] = all_data if all_data else None
            st.session_state['marker_point'] = marker if marker else None

    marker_point_to_display = st.session_state.get('marker_point', None)
    
    with st.spinner('Generating map...'):
        display_data, vmin, vmax = normalize_data(main_data_info['data'], st.session_state.norm_method)
        title = f'{st.session_state.element} Kd Distribution in {st.session_state.depth} Soil ({st.session_state.norm_method})'
        img_buf = None
        try:
            # First, try the advanced function
            img_buf = create_map_with_latlon(display_data, vmin, vmax, main_data_info, title, marker_point_to_display)
        except Exception as e:
            st.warning(f"Advanced map view failed: {e}. Displaying fallback map.", icon="⚠️")
            # If it fails, use the absolutely safe fallback
            try:
                img_buf = create_map_fallback(display_data, vmin, vmax, main_data_info, title)
            except Exception as fallback_e:
                st.error(f"Fallback map also failed: {fallback_e}")

    if img_buf: st.image(img_buf, use_column_width='always')
    else: st.error("Map generation failed completely.")

with col_right:
    all_query_data = st.session_state.get('all_query_data')
    tab1, tab2 = st.tabs(["📍 Query Results", "📈 Depth Profile Analysis"])

    with tab1:
        if all_query_data and st.session_state.depth in all_query_data:
            params = all_query_data[st.session_state.depth]
            st.success("✅ Query Successful")
            # ... (display logic is safe)
        else:
            if st.session_state.query_button: st.warning("⚠️ No valid data at this location.")
            else: st.info("👆 Enter coordinates and click 'Query Point'.")
            # ... (empty display logic is safe)

    with tab2:
        # Wrap the entire tab's logic in a try-except block
        try:
            st.header("Vertical Kd Distribution")
            if all_query_data:
                st.info("Depth profile data is available based on your latest query.")
                profile_chart = create_depth_profile_chart(all_query_data, st.session_state.element)
                if profile_chart: st.pyplot(profile_chart)
                else: st.warning("Could not create profile chart. Not enough data points.")
            else:
                st.warning("Please perform a successful query to view depth profile data.")
        except Exception as e:
            st.error(f"An error occurred in the Depth Profile tab: {e}", icon="🔥")

# --- v1.16 Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>🌱 REEs Soil Kd Visualization System v1.16<br>Fortress Edition with Fault Tolerance</div>", unsafe_allow_html=True)