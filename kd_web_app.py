import streamlit as st
import numpy as np
import rasterio
from rasterio.warp import transform as coord_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import pandas as pd
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ==================== 英文界面设置 ====================
# 设置matplotlib字体（使用英文默认字体）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(
    page_title="REE Soil Kd Value Visualization",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题
st.title("🌱 REE Soil Kd Value Visualization System")

# 设置数据目录 - 优化路径处理
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(current_dir, "data")

# 检查数据目录
if not os.path.exists(DATA_DIR):
    st.error(f"Data directory does not exist: {DATA_DIR}")
    st.info("Please place data files in the following directory: " + DATA_DIR)
    st.stop()

def wgs84_to_albers(lon, lat, crs):
    """Convert longitude and latitude to Albers coordinates"""
    try:
        x, y = coord_transform('EPSG:4326', crs, [lon], [lat])
        return x[0], y[0]
    except:
        return None, None

def create_enhanced_colormap():
    """Create enhanced contrast colormap"""
    colors = ['#00008B', '#0000FF', '#0080FF', '#00BFFF',
             '#00FF80', '#80FF00', '#FFFF00', '#FF8000',
             '#FF0000', '#8B0000']
    return LinearSegmentedColormap.from_list('enhanced_viridis', colors, N=256)

def normalize_data(data, method):
    """Normalize data"""
    data_copy = np.array(data, copy=True)
    
    if np.ma.is_masked(data):
        valid_data = data.compressed()
    else:
        valid_mask = np.isfinite(data_copy)
        valid_data = data_copy[valid_mask]
    
    if len(valid_data) == 0:
        return data_copy, 0, 1
    
    if method == "Raw Data":
        data_copy[data_copy < 0] = 0
        return data_copy, 0, np.max(valid_data)
    
    elif method == "Percentile Normalization":
        p5 = np.percentile(valid_data, 5)
        p95 = np.percentile(valid_data, 95)
        if p95 - p5 > 1e-10:
            normalized = (data_copy - p5) / (p95 - p5)
            normalized = np.clip(normalized, 0, 1)
            return normalized, 0, 1
        return data_copy, 0, 1
            
    elif method == "Standard Deviation Normalization":
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        if std > 1e-10:
            normalized = (data_copy - mean) / (2 * std) + 0.5
            normalized = np.clip(normalized, 0, 1)
            return normalized, 0, 1
        return data_copy, 0, 1
            
    elif method == "Linear Normalization":
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        if max_val - min_val > 1e-10:
            normalized = (data_copy - min_val) / (max_val - min_val)
            return normalized, 0, 1
        return data_copy, 0, 1
    
    return data_copy, np.min(valid_data), np.max(valid_data)

@st.cache_data
def load_raster_data(file_path):
    """Load raster data"""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1).astype(np.float32)
            transform_matrix = src.transform
            crs = src.crs
            bounds = src.bounds
            
            # Handle invalid values
            data[~np.isfinite(data)] = np.nan
            data = np.ma.masked_invalid(data)
            
            return {
                'data': data,
                'transform': transform_matrix,
                'crs': crs,
                'bounds': bounds
            }
    except Exception as e:
        st.error(f"Error loading raster data: {str(e)}")
        return None

def get_point_parameters(lon, lat, element, depth_suffix, data_info):
    """Get all parameters for a specific point"""
    try:
        x, y = wgs84_to_albers(lon, lat, data_info['crs'])
        if x is None or y is None:
            return None
        
        row, col = rasterio.transform.rowcol(data_info['transform'], x, y)
        
        if (row < 0 or row >= data_info['data'].shape[0] or 
            col < 0 or col >= data_info['data'].shape[1]):
            return None
        
        kd_value = data_info['data'][row, col]
        if np.ma.is_masked(kd_value) or not np.isfinite(kd_value):
            return None
        
        params = {"Kd": float(kd_value)}
        
        # Read other parameters
        param_files = {
            "pH": f"ph{depth_suffix}.tif",
            "SOM": f"soc{depth_suffix}.tif",
            "CEC": f"cec{depth_suffix}.tif",
            "Ce": f"{element}.tif"
        }
        
        for param_name, filename in param_files.items():
            file_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(file_path):
                try:
                    with rasterio.open(file_path) as src:
                        value = src.read(1)[row, col]
                        if param_name == "pH" or param_name == "CEC":
                            value = value / 100
                        elif param_name == "SOM":
                            value = value * 1.724 / 100
                        params[param_name] = float(value)
                except Exception as e:
                    st.warning(f"Failed to load {param_name}: {str(e)}")
        
        # Calculate IS
        ec_file = "T_ECE.tif" if depth_suffix in ["05", "515", "1530"] else "S_ECE.tif"
        ec_path = os.path.join(DATA_DIR, ec_file)
        if os.path.exists(ec_path):
            try:
                with rasterio.open(ec_path) as src:
                    ec_value = src.read(1)[row, col]
                    is_value = max(0.0446 * ec_value - 0.000173, 0)
                    params["IS"] = float(is_value)
            except Exception as e:
                st.warning(f"Failed to calculate IS: {str(e)}")
        
        return params, (row, col)
        
    except Exception as e:
        st.error(f"Error getting point parameters: {str(e)}")
        return None

def create_map_image(display_data, vmin, vmax, element, depth, norm_method, data_info, marker_point=None):
    """Create map image with optimized display"""
    try:
        # Create new figure
        fig = plt.figure(figsize=(12, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        # Choose colormap
        if norm_method == "Raw Data":
            cmap = 'viridis'
        else:
            cmap = create_enhanced_colormap()
        
        # ==================== Map Display Optimization ====================
        # Get data bounds
        bounds = data_info['bounds']
        
        # Calculate appropriate display range (reduce map scale, minimize whitespace)
        width = bounds.right - bounds.left
        height = bounds.top - bounds.bottom
        
        # Add appropriate margins (smaller than before)
        margin_x = width * 0.05  # Only 5% margin
        margin_y = height * 0.05
        
        # Set display range
        extent = [
            bounds.left - margin_x,
            bounds.right + margin_x, 
            bounds.bottom - margin_y,
            bounds.top + margin_y
        ]
        
        # Display data - use optimized range
        im = ax.imshow(
            display_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,  # Use optimized range
            aspect='auto',   # Auto adjust aspect ratio
            interpolation='nearest',
            origin='upper'
        )
        # ==================== Map Display Optimization End ====================
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Kd Value [L/g]', fontsize=10)
        
        # Set title
        title_text = f'{element} Kd Distribution in {depth} Soil ({norm_method})'
        ax.set_title(title_text, fontsize=12)
        
        # Set axis labels
        ax.set_xlabel('East Coordinate (m)', fontsize=10)
        ax.set_ylabel('North Coordinate (m)', fontsize=10)
        
        # Add grid - finer and lighter
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)
        
        # Add query point marker
        if marker_point is not None:
            row, col = marker_point
            # Calculate actual coordinates of marker point
            x, y = rasterio.transform.xy(data_info['transform'], row, col)
            ax.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
            
            label_text = 'Query Point'
                
            ax.annotate(
                label_text,
                xy=(x, y),
                xytext=(x + width * 0.02, y - height * 0.02),
                fontsize=9,
                color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
            )
        
        # Adjust layout - more compact
        plt.tight_layout(pad=2.0)
        
        # Save to byte stream
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Close figure
        plt.close(fig)
        
        return buf
    except Exception as e:
        st.error(f"Error creating map image: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.header("📊 Parameter Settings")
    
    element = st.selectbox(
        "Rare Earth Element",
        ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y"],
        help="Select rare earth element to display"
    )
    
    depth = st.selectbox(
        "Soil Depth",
        ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"],
        help="Select soil sampling depth"
    )
    
    norm_method = st.selectbox(
        "Normalization Method",
        ["Raw Data", "Percentile Normalization", "Standard Deviation Normalization", "Linear Normalization"],
        help="Select data normalization method"
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
    show_stats = st.checkbox("Show Statistics", value=False)

# Depth mapping
depth_mapping = {
    "0-5cm": "05", "5-15cm": "515", "15-30cm": "1530",
    "30-60cm": "3060", "60-100cm": "60100"
}
depth_suffix = depth_mapping[depth]

# File path
raster_filename = f"prediction_result_{element}{depth_suffix}_raw.tif"
raster_path = os.path.join(DATA_DIR, raster_filename)

# Create two-column layout
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Kd Value Spatial Distribution")
    
    if not os.path.exists(raster_path):
        st.error(f"❌ File not found: {raster_filename}")
        st.info("Please check if data files exist")
        st.stop()
    
    # Load data
    with st.spinner('Loading data...'):
        data_info = load_raster_data(raster_path)
    
    if data_info is None:
        st.error("Unable to load data file")
        st.stop()
    
    # Show statistics
    if show_stats:
        valid_data = data_info['data'].compressed() if np.ma.is_masked(data_info['data']) else data_info['data'].flatten()
        valid_data = valid_data[np.isfinite(valid_data)]
        
        if len(valid_data) > 0:
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Min", f"{np.min(valid_data):.4f}")
            with col_stat2:
                st.metric("Max", f"{np.max(valid_data):.4f}")
            with col_stat3:
                st.metric("Mean", f"{np.mean(valid_data):.4f}")
            with col_stat4:
                st.metric("Median", f"{np.median(valid_data):.4f}")
    
    # Data processing and display
    try:
        # Normalization
        display_data, vmin, vmax = normalize_data(data_info['data'], norm_method)
        
        # Handle query
        marker_point = None
        if query_button:
            result = get_point_parameters(lon, lat, element, depth_suffix, data_info)
            if result:
                params, marker_point = result
                st.session_state['query_result'] = params
            else:
                st.session_state['query_result'] = None
        
        # Generate map image - pass data_info for optimized display
        with st.spinner('Generating map...'):
            img_buf = create_map_image(display_data, vmin, vmax, element, depth, norm_method, data_info, marker_point)
            
        if img_buf:
            # Display image
            st.image(img_buf, use_container_width=True)
        else:
            st.error("Failed to generate map image")
        
    except Exception as e:
        st.error(f"Map generation error: {str(e)}")

with col_right:
    st.subheader("📍 Query Results")
    
    # Check if there are query results
    if 'query_result' in st.session_state and st.session_state['query_result'] is not None:
        params = st.session_state['query_result']
        
        st.success("✅ Query Successful")
        
        # Location information
        st.markdown("**📍 Location Information**")
        st.write(f"- Longitude: {lon:.4f}°E")
        st.write(f"- Latitude: {lat:.4f}°N")
        st.write(f"- Element: {element}")
        st.write(f"- Depth: {depth}")
        
        st.markdown("---")
        
        # Parameter table
        st.markdown("**📊 Soil Parameters**")
        
        param_display = []
        param_info = {
            "Kd": ("L/g", "Distribution Coefficient"),
            "pH": ("", "Soil Acidity/Alkalinity"),
            "SOM": ("g/kg", "Organic Matter Content"),
            "CEC": ("cmol⁺/kg", "Cation Exchange Capacity"),
            "IS": ("mol/L", "Ionic Strength"),
            "Ce": ("mg/kg", "Equilibrium Concentration")
        }
        
        for param_name in ["Kd", "pH", "SOM", "CEC", "IS", "Ce"]:
            if param_name in params:
                value = params[param_name]
                unit, desc = param_info[param_name]
                value_str = f"{value:.2f}" if value >= 1 else f"{value:.4f}"
                param_display.append({
                    "Parameter": param_name,
                    "Value": value_str,
                    "Unit": unit
                })
        
        df = pd.DataFrame(param_display)
        st.dataframe(df, hide_index=True, use_container_width=True)
        
        # Parameter description
        with st.expander("📖 Parameter Description"):
            st.markdown("""
            - **Kd**: Distribution coefficient, represents element distribution between solid and liquid phases
            - **pH**: Soil acidity/alkalinity
            - **SOM**: Soil organic matter content
            - **CEC**: Cation exchange capacity
            - **IS**: Ionic strength
            - **Ce**: Equilibrium concentration
            """)
    else:
        if query_button:
            st.warning("⚠️ No valid data at this location or out of range")
        else:
            st.info("👆 Enter coordinates and click query button")
        
        # Show empty table
        st.markdown("**📊 Soil Parameters**")
        empty_df = pd.DataFrame({
            "Parameter": ["Kd", "pH", "SOM", "CEC", "IS", "Ce"],
            "Value": ["--"] * 6,
            "Unit": ["L/g", "", "g/kg", "cmol⁺/kg", "mol/L", "mg/kg"]
        })
        st.dataframe(empty_df, hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    🌱 REE Soil Kd Visualization System v2.0 | English Interface<br>
    Data based on Albers Equal Area Conic Projection | Supports 15 rare earth elements
</div>
""", unsafe_allow_html=True)