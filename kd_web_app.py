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
from huggingface_hub import hf_hub_download # <--- 修改点 1: 导入新库

warnings.filterwarnings('ignore')

# ==================== Hugging Face 仓库配置 ====================
# 【【【极其重要】】】请将 "YourUsername" 替换成您的 Hugging Face 用户名！
HF_REPO_ID = "Tong-DAO/REES"
HF_REPO_TYPE = "dataset"

# ==================== 英文界面设置 ====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="REE Soil Kd Value Visualization",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌱 REE Soil Kd Value Visualization System")

# <--- 修改点 2: 移除所有关于本地 "data" 文件夹的代码。不再需要 DATA_DIR。

# ==================== 数据加载函数（已修改） ====================

@st.cache_resource(show_spinner="正在从云端同步数据文件...")
def get_hf_file_path(filename_in_repo):
    """
    从Hugging Face Hub下载文件并返回其本地缓存路径。
    使用 st.cache_resource 确保文件只下载一次。
    """
    try:
        # 文件在仓库中的完整路径是 'data/文件名'
        full_path_in_repo = f"data/{filename_in_repo}"
        
        # 从Streamlit secrets获取token，这是在云端部署时的标准做法
        token = st.secrets.get("HUGGING_FACE_HUB_TOKEN")

        return hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=full_path_in_repo,
            repo_type=HF_REPO_TYPE,
            use_auth_token=token
        )
    except Exception as e:
        # 如果下载失败，显示更详细的错误
        st.error(f"从Hugging Face下载文件 '{filename_in_repo}' 失败。错误: {e}")
        st.error(f"请检查：1. 仓库ID '{HF_REPO_ID}' 是否正确。 2. 文件 '{full_path_in_repo}' 是否存在于仓库中。 3. 是否在Streamlit Cloud中正确设置了HUGGING_FACE_HUB_TOKEN。")
        st.stop()


@st.cache_data(show_spinner="正在加载和解析栅格数据...")
def load_raster_data(filename_in_repo):
    """
    加载栅格数据。函数现在接收仓库中的文件名，而不是本地完整路径。
    """
    # <--- 修改点 3: 先获取文件的本地路径，再用 rasterio 打开
    local_file_path = get_hf_file_path(filename_in_repo)
    
    try:
        with rasterio.open(local_file_path) as src:
            data = src.read(1).astype(np.float32)
            transform_matrix = src.transform
            crs = src.crs
            bounds = src.bounds
            
            data[~np.isfinite(data)] = np.nan
            data = np.ma.masked_invalid(data)
            
            return {
                'data': data,
                'transform': transform_matrix,
                'crs': crs,
                'bounds': bounds
            }
    except Exception as e:
        st.error(f"使用Rasterio加载文件 '{local_file_path}' 时出错: {str(e)}")
        return None

# ==================== 其他函数（部分已修改） ====================

def wgs84_to_albers(lon, lat, crs):
    try:
        x, y = coord_transform('EPSG:4326', crs, [lon], [lat])
        return x[0], y[0]
    except:
        return None, None

def create_enhanced_colormap():
    colors = ['#00008B', '#0000FF', '#0080FF', '#00BFFF', '#00FF80', '#80FF00', '#FFFF00', '#FF8000', '#FF0000', '#8B0000']
    return LinearSegmentedColormap.from_list('enhanced_viridis', colors, N=256)

def normalize_data(data, method):
    data_copy = np.array(data, copy=True)
    if np.ma.is_masked(data): valid_data = data.compressed()
    else: valid_data = data_copy[np.isfinite(data_copy)]
    if len(valid_data) == 0: return data_copy, 0, 1
    if method == "Raw Data":
        data_copy[data_copy < 0] = 0
        return data_copy, 0, np.max(valid_data)
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


def get_point_parameters(lon, lat, element, depth_suffix, data_info):
    """
    获取单点所有参数。此函数现在会按需从Hugging Face下载每个参数文件。
    """
    try:
        x, y = wgs84_to_albers(lon, lat, data_info['crs'])
        if x is None: return None
        
        row, col = rasterio.transform.rowcol(data_info['transform'], x, y)
        
        if not (0 <= row < data_info['data'].shape[0] and 0 <= col < data_info['data'].shape[1]):
            return None
        
        kd_value = data_info['data'][row, col]
        if np.ma.is_masked(kd_value) or not np.isfinite(kd_value):
            return None
        
        params = {"Kd": float(kd_value)}
        
        param_files = {
            "pH": f"ph{depth_suffix}.tif", "SOM": f"soc{depth_suffix}.tif",
            "CEC": f"cec{depth_suffix}.tif", "Ce": f"{element}.tif"
        }
        
        for param_name, filename in param_files.items():
            # <--- 修改点 4: 按需下载每个参数文件
            try:
                file_path = get_hf_file_path(filename)
                with rasterio.open(file_path) as src:
                    value = src.read(1)[row, col]
                    if param_name in ["pH", "CEC"]: value /= 100
                    elif param_name == "SOM": value = value * 1.724 / 100
                    params[param_name] = float(value)
            except Exception:
                # 如果某个辅助文件下载或读取失败，只发出警告，不中断程序
                st.toast(f"无法加载辅助参数: {param_name}", icon="⚠️")
        
        ec_file = "T_ECE.tif" if depth_suffix in ["05", "515", "1530"] else "S_ECE.tif"
        try:
            # <--- 修改点 5: 下载EC文件以计算IS
            ec_path = get_hf_file_path(ec_file)
            with rasterio.open(ec_path) as src:
                ec_value = src.read(1)[row, col]
                is_value = max(0.0446 * ec_value - 0.000173, 0)
                params["IS"] = float(is_value)
        except Exception:
            st.toast("无法计算离子强度 (IS)", icon="⚠️")
            
        return params, (row, col)
        
    except Exception as e:
        st.error(f"获取点参数时发生错误: {str(e)}")
        return None

# ... create_map_image 函数保持不变，因为它是纯计算和绘图 ...
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
        ax.set_xlabel('East Coordinate (m)', fontsize=10); ax.set_ylabel('North Coordinate (m)', fontsize=10)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)
        if marker_point:
            row, col = marker_point
            x, y = rasterio.transform.xy(data_info['transform'], row, col)
            ax.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
            ax.annotate('Query Point', xy=(x, y), xytext=(x + width * 0.02, y - height * 0.02), fontsize=9, color='red', arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        plt.tight_layout(pad=2.0)
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf
    except Exception as e:
        st.error(f"创建地图图像时出错: {str(e)}")
        return None


# ==================== 主程序界面 ====================

with st.sidebar:
    st.header("📊 Parameter Settings")
    element = st.selectbox("Rare Earth Element", ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y"], help="Select rare earth element to display")
    depth = st.selectbox("Soil Depth", ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"], help="Select soil sampling depth")
    norm_method = st.selectbox("Normalization Method", ["Raw Data", "Percentile Normalization", "Standard Deviation Normalization", "Linear Normalization"], help="Select data normalization method")
    st.markdown("---")
    st.header("🔍 Coordinate Query")
    lon = st.number_input("Longitude", min_value=73.0, max_value=135.0, value=105.0, step=0.1)
    lat = st.number_input("Latitude", min_value=18.0, max_value=53.0, value=35.0, step=0.1)
    query_button = st.button("🎯 Query Point", use_container_width=True, type="primary")
    st.markdown("---")
    show_stats = st.checkbox("Show Statistics", value=False)

depth_mapping = {"0-5cm": "05", "5-15cm": "515", "15-30cm": "1530", "30-60cm": "3060", "60-100cm": "60100"}
depth_suffix = depth_mapping[depth]

raster_filename = f"prediction_result_{element}{depth_suffix}_raw.tif"
# <--- 修改点 6: 移除了 raster_path 和 os.path.exists 检查，因为文件现在从云端获取

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Kd Value Spatial Distribution")
    
    # <--- 修改点 7: 直接调用 load_raster_data 并传入文件名
    # spinner 会在 load_raster_data 和 get_hf_file_path 中自动显示
    data_info = load_raster_data(raster_filename)
    
    if data_info is None:
        st.error("无法加载数据文件。请刷新页面重试。")
        st.stop()
    
    if show_stats:
        valid_data = data_info['data'].compressed() if np.ma.is_masked(data_info['data']) else data_info['data'].flatten()
        valid_data = valid_data[np.isfinite(valid_data)]
        if len(valid_data) > 0:
            stats_cols = st.columns(4)
            stats_cols[0].metric("Min", f"{np.min(valid_data):.4f}")
            stats_cols[1].metric("Max", f"{np.max(valid_data):.4f}")
            stats_cols[2].metric("Mean", f"{np.mean(valid_data):.4f}")
            stats_cols[3].metric("Median", f"{np.median(valid_data):.4f}")
    
    try:
        display_data, vmin, vmax = normalize_data(data_info['data'], norm_method)
        marker_point = None
        if query_button:
            result = get_point_parameters(lon, lat, element, depth_suffix, data_info)
            if result:
                st.session_state['query_result'], marker_point = result
            else:
                st.session_state['query_result'] = None
        
        with st.spinner('正在生成地图...'):
            img_buf = create_map_image(display_data, vmin, vmax, element, depth, norm_method, data_info, marker_point)
            
        if img_buf:
            st.image(img_buf, use_container_width=True)
        else:
            st.error("无法生成地图图像。")
        
    except Exception as e:
        st.error(f"地图生成时发生错误: {str(e)}")

with col_right:
    st.subheader("📍 Query Results")
    
    if 'query_result' in st.session_state and st.session_state['query_result'] is not None:
        params = st.session_state['query_result']
        st.success("✅ Query Successful")
        st.markdown(f"**📍 Location Information**\n- Longitude: {lon:.4f}°E\n- Latitude: {lat:.4f}°N\n- Element: {element}\n- Depth: {depth}")
        st.markdown("---")
        st.markdown("**📊 Soil Parameters**")
        
        param_display = []
        param_info = {"Kd": ("L/g", "Distribution Coefficient"), "pH": ("", "Soil Acidity/Alkalinity"), "SOM": ("g/kg", "Organic Matter Content"), "CEC": ("cmol⁺/kg", "Cation Exchange Capacity"), "IS": ("mol/L", "Ionic Strength"), "Ce": ("mg/kg", "Equilibrium Concentration")}
        
        for param_name in ["Kd", "pH", "SOM", "CEC", "IS", "Ce"]:
            if param_name in params:
                value = params[param_name]
                unit, desc = param_info[param_name]
                param_display.append({"Parameter": param_name, "Value": f"{value:.2f}" if value >= 1 else f"{value:.4f}", "Unit": unit})
        
        st.dataframe(pd.DataFrame(param_display), hide_index=True, use_container_width=True)
        
        with st.expander("📖 Parameter Description"):
            st.markdown("- **Kd**: Distribution coefficient\n- **pH**: Soil acidity/alkalinity\n- **SOM**: Soil organic matter content\n- **CEC**: Cation exchange capacity\n- **IS**: Ionic strength\n- **Ce**: Equilibrium concentration")
    else:
        if query_button: st.warning("⚠️ No valid data at this location or out of range")
        else: st.info("👆 Enter coordinates and click query button")
        st.markdown("**📊 Soil Parameters**")
        empty_df = pd.DataFrame({"Parameter": ["Kd", "pH", "SOM", "CEC", "IS", "Ce"], "Value": ["--"] * 6, "Unit": ["L/g", "", "g/kg", "cmol⁺/kg", "mol/L", "mg/kg"]})
        st.dataframe(empty_df, hide_index=True, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 12px;'>🌱 REE Soil Kd Visualization System v2.1 | English Interface<br>Data hosted on Hugging Face Hub | Fetches data on demand</div>", unsafe_allow_html=True)