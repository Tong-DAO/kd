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
import os
from huggingface_hub import hf_hub_download

warnings.filterwarnings('ignore')

# ==================== Configuration ====================
HF_REPO_ID = "Tong-DAO/REES"
HF_REPO_TYPE = "dataset"
DATA_DIR = "data"  # Local data directory

# ==================== English Interface Settings ====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="REE Soil Kd Value Visualization",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌱 REE Soil Kd Value Visualization System")

# ==================== Enhanced Data Loading Functions ====================

@st.cache_resource(show_spinner="Downloading data files from Hugging Face Hub...")
def get_hf_file_path(filename_in_repo):
    """
    Enhanced file download with better error handling and API compatibility
    """
    try:
        full_path_in_repo = f"{DATA_DIR}/{filename_in_repo}"
        
        # Get token from secrets (handle both naming conventions)
        token = st.secrets.get("HUGGING_FACE_HUB_TOKEN") or st.secrets.get("HUGGINGFACE_TOKEN")
        
        # Use modern API - token parameter instead of use_auth_token
        download_kwargs = {
            "repo_id": HF_REPO_ID,
            "filename": full_path_in_repo,
            "repo_type": HF_REPO_TYPE,
        }
        
        # Only add token if it exists
        if token:
            download_kwargs["token"] = token
        
        local_path = hf_hub_download(**download_kwargs)
        
        # Verify file exists and has content
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path
        else:
            raise FileNotFoundError(f"Downloaded file is empty or missing: {local_path}")
            
    except Exception as e:
        st.error(f"❌ Failed to download file '{filename_in_repo}' from Hugging Face Hub")
        st.error(f"Error details: {str(e)}")
        st.info("""
        **Troubleshooting steps:**
        1. Check if repository '{HF_REPO_ID}' exists and is accessible
        2. Verify file 'data/{filename_in_repo}' exists in the repository
        3. Ensure HUGGING_FACE_HUB_TOKEN is set in Streamlit Cloud secrets
        4. Try refreshing the page to reattempt download
        """)
        st.stop()

@st.cache_data(show_spinner="Processing raster data...")
def load_raster_data(filename_in_repo):
    """
    Enhanced raster loading with better memory management
    """
    try:
        local_file_path = get_hf_file_path(filename_in_repo)
        
        with rasterio.open(local_file_path) as src:
            # Read data with memory optimization
            data = src.read(1).astype(np.float32)
            
            # Handle invalid values more efficiently
            data = np.ma.masked_invalid(data)
            data.fill_value = np.nan
            
            return {
                'data': data,
                'transform': src.transform,
                'crs': src.crs,
                'bounds': src.bounds,
                'shape': data.shape
            }
            
    except Exception as e:
        st.error(f"❌ Error loading raster file: {str(e)}")
        return None

# ==================== Core Utility Functions ====================

def wgs84_to_albers(lon, lat, crs):
    """Convert WGS84 to Albers projection with error handling"""
    try:
        x, y = coord_transform('EPSG:4326', crs, [lon], [lat])
        return float(x[0]), float(y[0])
    except Exception:
        return None, None

def create_enhanced_colormap():
    """Create optimized colormap for better visualization"""
    colors = [
        '#00008B', '#0000FF', '#0080FF', '#00BFFF', '#00FF80',
        '#80FF00', '#FFFF00', '#FF8000', '#FF0000', '#8B0000'
    ]
    return LinearSegmentedColormap.from_list('enhanced_spectral', colors, N=256)

def normalize_data(data, method):
    """
    Enhanced normalization with better numerical stability
    """
    if np.ma.is_masked(data):
        valid_data = data.compressed()
    else:
        valid_mask = np.isfinite(data)
        valid_data = data[valid_mask]
    
    if len(valid_data) == 0:
        return data, 0, 1
    
    data_copy = np.ma.array(data, copy=True, fill_value=np.nan)
    
    try:
        if method == "Raw Data":
            # Clip negative values for physical meaning
            data_copy = np.ma.maximum(data_copy, 0)
            vmax = np.percentile(valid_data, 99.5)  # Use 99.5% to avoid outliers
            return data_copy, 0, vmax
            
        elif method == "Percentile Normalization":
            p2 = np.percentile(valid_data, 2)
            p98 = np.percentile(valid_data, 98)
            if p98 - p2 > 1e-10:
                normalized = (data_copy - p2) / (p98 - p2)
                normalized = np.ma.clip(normalized, 0, 1)
                return normalized, 0, 1
                
        elif method == "Standard Deviation Normalization":
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            if std > 1e-10:
                normalized = (data_copy - mean) / (2 * std) + 0.5
                normalized = np.ma.clip(normalized, 0, 1)
                return normalized, 0, 1
                
        elif method == "Linear Normalization":
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)
            if max_val - min_val > 1e-10:
                normalized = (data_copy - min_val) / (max_val - min_val)
                return normalized, 0, 1
                
    except Exception as e:
        st.warning(f"Normalization warning: {e}. Using raw data.")
    
    # Fallback to raw data with safe limits
    vmax = np.percentile(valid_data, 99) if len(valid_data) > 0 else 1
    return data_copy, 0, vmax

def get_point_parameters(lon, lat, element, depth_suffix, data_info):
    """
    Enhanced point query with better parameter handling
    """
    try:
        # Convert coordinates
        x, y = wgs84_to_albers(lon, lat, data_info['crs'])
        if x is None or y is None:
            return None, None
        
        # Get raster position
        row, col = rasterio.transform.rowcol(data_info['transform'], x, y)
        
        # Validate bounds
        if not (0 <= row < data_info['shape'][0] and 0 <= col < data_info['shape'][1]):
            return None, None
        
        # Extract Kd value
        kd_value = data_info['data'][row, col]
        if np.ma.is_masked(kd_value) or not np.isfinite(kd_value):
            return None, None
        
        params = {"Kd": float(kd_value)}
        
        # Load auxiliary parameters
        param_files = {
            "pH": f"ph{depth_suffix}.tif",
            "SOM": f"soc{depth_suffix}.tif", 
            "CEC": f"cec{depth_suffix}.tif",
            "Ce": f"{element}.tif"
        }
        
        for param_name, filename in param_files.items():
            try:
                file_path = get_hf_file_path(filename)
                with rasterio.open(file_path) as src:
                    value = src.read(1)[row, col]
                    # Apply parameter-specific transformations
                    if param_name == "pH":
                        value = value / 100.0  # pH scaling
                    elif param_name == "CEC":
                        value = value / 100.0  # CEC scaling
                    elif param_name == "SOM":
                        value = value * 1.724 / 100.0  # SOC to SOM conversion
                    
                    if np.isfinite(value):
                        params[param_name] = float(value)
            except Exception as e:
                st.toast(f"⚠️ Could not load {param_name}: {str(e)}", icon="⚠️")
        
        # Calculate Ionic Strength
        ec_file = "T_ECE.tif" if depth_suffix in ["05", "515", "1530"] else "S_ECE.tif"
        try:
            ec_path = get_hf_file_path(ec_file)
            with rasterio.open(ec_path) as src:
                ec_value = src.read(1)[row, col]
                if np.isfinite(ec_value):
                    is_value = max(0.0446 * ec_value - 0.000173, 0)
                    params["IS"] = float(is_value)
        except Exception:
            st.toast("⚠️ Could not calculate Ionic Strength", icon="⚠️")
        
        return params, (row, col)
        
    except Exception as e:
        st.error(f"❌ Error in point query: {str(e)}")
        return None, None

def create_map_image(display_data, vmin, vmax, element, depth, norm_method, data_info, marker_point=None):
    """
    Enhanced map generation with better visualization
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100, facecolor='white')
        
        # Choose colormap based on normalization
        if norm_method == "Raw Data":
            cmap = 'viridis'
        else:
            cmap = create_enhanced_colormap()
        
        # Calculate optimized display extent
        bounds = data_info['bounds']
        width = bounds.right - bounds.left
        height = bounds.top - bounds.bottom
        margin = 0.02  # 2% margin
        extent = [
            bounds.left - width * margin,
            bounds.right + width * margin,
            bounds.bottom - height * margin, 
            bounds.top + height * margin
        ]
        
        # Create image with optimized settings
        im = ax.imshow(
            display_data, 
            cmap=cmap, 
            vmin=vmin, 
            vmax=vmax,
            extent=extent,
            aspect='auto',
            interpolation='nearest',
            origin='upper'
        )
        
        # Enhanced colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar.set_label('Kd Value [L/g]', fontsize=11, weight='bold')
        
        # Title and labels
        ax.set_title(
            f'{element} Kd Distribution - {depth} Depth ({norm_method})', 
            fontsize=14, 
            pad=20,
            weight='bold'
        )
        ax.set_xlabel('East Coordinate (m)', fontsize=11)
        ax.set_ylabel('North Coordinate (m)', fontsize=11)
        
        # Add subtle grid
        ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
        
        # Mark query point if provided
        if marker_point:
            row, col = marker_point
            x, y = rasterio.transform.xy(data_info['transform'], row, col)
            ax.plot(x, y, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
            ax.annotate(
                'Query Point', 
                xy=(x, y), 
                xytext=(15, 15),
                textcoords='offset points',
                fontsize=10,
                color='red',
                weight='bold',
                arrowprops=dict(
                    arrowstyle='->', 
                    color='red', 
                    lw=2,
                    connectionstyle='arc3,rad=0.1'
                )
            )
        
        plt.tight_layout(pad=2.0)
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                   facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        plt.close(fig)
        
        return buf
        
    except Exception as e:
        st.error(f"❌ Map generation error: {str(e)}")
        return None

# ==================== Main Application Interface ====================

# Sidebar configuration
with st.sidebar:
    st.header("📊 Parameter Settings")
    
    element = st.selectbox(
        "Rare Earth Element",
        ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y"],
        help="Select rare earth element for analysis"
    )
    
    depth = st.selectbox(
        "Soil Depth", 
        ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"],
        help="Select soil sampling depth"
    )
    
    norm_method = st.selectbox(
        "Normalization Method",
        ["Raw Data", "Percentile Normalization", "Standard Deviation Normalization", "Linear Normalization"],
        help="Select data normalization approach"
    )
    
    st.markdown("---")
    st.header("🔍 Coordinate Query")
    
    col1, col2 = st.columns(2)
    with col1:
        lon = st.number_input("Longitude", min_value=73.0, max_value=135.0, value=105.0, step=0.1)
    with col2:
        lat = st.number_input("Latitude", min_value=18.0, max_value=53.0, value=35.0, step=0.1)
    
    query_button = st.button("🎯 Query Point", use_container_width=True, type="primary")
    
    st.markdown("---")
    show_stats = st.checkbox("Show Statistics", value=True)

# Depth mapping and file preparation
depth_mapping = {
    "0-5cm": "05", "5-15cm": "515", "15-30cm": "1530",
    "30-60cm": "3060", "60-100cm": "60100"
}
depth_suffix = depth_mapping[depth]
raster_filename = f"prediction_result_{element}{depth_suffix}_raw.tif"

# Main content layout
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Spatial Distribution of Kd Values")
    
    # Load and validate data
    with st.spinner('🔄 Loading spatial data...'):
        data_info = load_raster_data(raster_filename)
    
    if data_info is None:
        st.error("❌ Unable to load required data file. Please check the file availability.")
        st.stop()
    
    # Display statistics if enabled
    if show_stats:
        valid_data = data_info['data'].compressed()
        if len(valid_data) > 0:
            st.metric("Data Points", f"{len(valid_data):,}")
            cols = st.columns(4)
            cols[0].metric("Minimum", f"{np.min(valid_data):.4f}")
            cols[1].metric("Maximum", f"{np.max(valid_data):.4f}") 
            cols[2].metric("Mean", f"{np.mean(valid_data):.4f}")
            cols[3].metric("Std Dev", f"{np.std(valid_data):.4f}")
    
    # Process and display map
    try:
        display_data, vmin, vmax = normalize_data(data_info['data'], norm_method)
        marker_point = None
        
        # Handle point query
        if query_button:
            with st.spinner('🔍 Querying location...'):
                result = get_point_parameters(lon, lat, element, depth_suffix, data_info)
                if result:
                    st.session_state['query_result'], marker_point = result
                    st.success("✅ Location query successful!")
                else:
                    st.session_state['query_result'] = None
                    st.warning("⚠️ No data available at specified coordinates")
        
        # Generate map
        with st.spinner('🎨 Generating visualization...'):
            img_buf = create_map_image(display_data, vmin, vmax, element, depth, norm_method, data_info, marker_point)
        
        if img_buf:
            st.image(img_buf, use_container_width=True, caption=f"Kd Distribution Map - {element} at {depth}")
        else:
            st.error("❌ Failed to generate map visualization")
            
    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")

with col_right:
    st.subheader("📍 Query Results")
    
    if 'query_result' in st.session_state and st.session_state['query_result'] is not None:
        params = st.session_state['query_result']
        
        # Location information
        st.success("✅ Query Successful")
        st.markdown("**📍 Location Details**")
        st.write(f"• **Longitude**: {lon:.4f}°E")
        st.write(f"• **Latitude**: {lat:.4f}°N") 
        st.write(f"• **Element**: {element}")
        st.write(f"• **Depth**: {depth}")
        
        st.markdown("---")
        
        # Parameter table
        st.markdown("**📊 Soil Parameters**")
        param_info = {
            "Kd": ("L/g", "Distribution Coefficient"),
            "pH": ("", "Soil Acidity/Alkalinity"),
            "SOM": ("g/kg", "Organic Matter Content"),
            "CEC": ("cmol⁺/kg", "Cation Exchange Capacity"),
            "IS": ("mol/L", "Ionic Strength"),
            "Ce": ("mg/kg", "Equilibrium Concentration")
        }
        
        param_data = []
        for param_name, (unit, desc) in param_info.items():
            if param_name in params:
                value = params[param_name]
                value_str = f"{value:.3f}" if abs(value) >= 1 else f"{value:.4f}"
                param_data.append({"Parameter": param_name, "Value": value_str, "Unit": unit})
        
        if param_data:
            st.dataframe(pd.DataFrame(param_data), hide_index=True, use_container_width=True)
        
        # Parameter descriptions
        with st.expander("📖 Parameter Descriptions"):
            st.markdown("""
            - **Kd**: Distribution coefficient representing element partitioning between solid and liquid phases
            - **pH**: Measure of soil acidity or alkalinity
            - **SOM**: Soil organic matter content (converted from SOC)
            - **CEC**: Cation exchange capacity of the soil
            - **IS**: Ionic strength of soil solution
            - **Ce**: Equilibrium concentration in soil solution
            """)
            
    else:
        if query_button:
            st.warning("⚠️ No valid data found at the specified location")
        else:
            st.info("👆 Enter coordinates and click 'Query Point' to analyze specific locations")
        
        # Placeholder table
        st.markdown("**📊 Soil Parameters**")
        empty_data = {
            "Parameter": ["Kd", "pH", "SOM", "CEC", "IS", "Ce"],
            "Value": ["--"] * 6,
            "Unit": ["L/g", "", "g/kg", "cmol⁺/kg", "mol/L", "mg/kg"]
        }
        st.dataframe(pd.DataFrame(empty_data), hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    🌱 REE Soil Kd Visualization System v2.3 | Enhanced English Interface<br>
    Data hosted on Hugging Face Hub | Real-time spatial analysis
</div>
""", unsafe_allow_html=True)

# Clear session state on page refresh
if st.button("🔄 Reset Application", use_container_width=True):
    st.session_state.clear()
    st.rerun()