import streamlit as st
import numpy as np
import rasterio
from rasterio.warp import transform as coord_transform
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
from matplotlib.colors import LinearSegmentedColormap
import traceback
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(
    page_title="稀土元素土壤Kd值可视化",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题
st.title("🌱 稀土元素土壤Kd值可视化")

# 设置数据目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# 检查数据目录
if not os.path.exists(DATA_DIR):
    st.error(f"数据目录不存在: {DATA_DIR}")
    st.info("请将数据文件放置在以下目录: " + DATA_DIR)
    st.stop()

# 初始化session state
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

def get_albers_projection():
    """创建Albers投影（不使用缓存）"""
    return ccrs.AlbersEqualArea(
        central_longitude=105,
        standard_parallels=(25, 47),
        false_easting=0,
        false_northing=0,
        globe=ccrs.Globe(datum="WGS84")
    )

def wgs84_to_albers(lon, lat, crs):
    """将经纬度转换为Albers坐标"""
    try:
        x, y = coord_transform('EPSG:4326', crs, [lon], [lat])
        return x[0], y[0]
    except:
        return None, None

def create_enhanced_colormap():
    """创建增强对比度的颜色映射"""
    colors = ['#00008B', '#0000FF', '#0080FF', '#00BFFF',
             '#00FF80', '#80FF00', '#FFFF00', '#FF8000',
             '#FF0000', '#8B0000']
    return LinearSegmentedColormap.from_list('enhanced_viridis', colors, N=256)

def normalize_data(data, method):
    """对数据进行归一化处理"""
    # 创建数据副本
    data_copy = np.array(data, copy=True)
    
    if np.ma.is_masked(data):
        valid_mask = ~data.mask
        valid_data = data.compressed()
    else:
        valid_mask = np.isfinite(data_copy)
        valid_data = data_copy[valid_mask]
    
    if len(valid_data) == 0:
        return data_copy, 0, 1
    
    if method == "原始数据":
        # 将负值设为0
        data_copy[data_copy < 0] = 0
        return data_copy, 0, np.max(valid_data)
    
    elif method == "百分位数归一化":
        p5 = np.percentile(valid_data, 5)
        p95 = np.percentile(valid_data, 95)
        if p95 - p5 > 1e-10:
            normalized = (data_copy - p5) / (p95 - p5)
            normalized = np.clip(normalized, 0, 1)
            return normalized, 0, 1
            
    elif method == "标准差归一化":
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        if std > 1e-10:
            normalized = (data_copy - mean) / (2 * std) + 0.5
            normalized = np.clip(normalized, 0, 1)
            return normalized, 0, 1
            
    elif method == "线性归一化":
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        if max_val - min_val > 1e-10:
            normalized = (data_copy - min_val) / (max_val - min_val)
            return normalized, 0, 1
    
    return data_copy, np.min(valid_data), np.max(valid_data)

@st.cache_data
def load_raster_data(file_path):
    """加载栅格数据"""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1).astype(np.float32)
            transform_matrix = src.transform
            crs = src.crs
            
            # 处理无效值
            data[~np.isfinite(data)] = np.nan
            data = np.ma.masked_invalid(data)
            
            return {
                'data': data,
                'transform': transform_matrix,
                'crs': crs
            }
    except Exception as e:
        st.error(f"加载文件失败: {str(e)}")
        return None

def get_point_parameters(lon, lat, element, depth_suffix, data_info):
    """获取指定点的所有参数值"""
    try:
        # 转换坐标
        x, y = wgs84_to_albers(lon, lat, data_info['crs'])
        if x is None or y is None:
            return None
        
        # 计算栅格索引
        row, col = rasterio.transform.rowcol(data_info['transform'], x, y)
        
        # 检查范围
        if (row < 0 or row >= data_info['data'].shape[0] or 
            col < 0 or col >= data_info['data'].shape[1]):
            return None
        
        # 获取Kd值
        kd_value = data_info['data'][row, col]
        if np.ma.is_masked(kd_value) or not np.isfinite(kd_value):
            return None
        
        params = {"Kd": float(kd_value)}
        
        # 定义参数文件
        param_files = {
            "pH": (f"ph{depth_suffix}.tif", 100),
            "SOM": (f"soc{depth_suffix}.tif", 1.724/100),
            "CEC": (f"cec{depth_suffix}.tif", 100),
            "Ce": (f"{element}.tif", 1)
        }
        
        # 读取其他参数
        for param_name, (filename, scale) in param_files.items():
            file_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(file_path):
                try:
                    with rasterio.open(file_path) as src:
                        value = src.read(1)[row, col]
                        if param_name == "SOM":
                            value = value * 1.724 / 100
                        elif scale != 1:
                            value = value / scale
                        params[param_name] = float(value)
                except:
                    pass
        
        # 读取EC值并计算IS
        ec_file = "T_ECE.tif" if depth_suffix in ["05", "515", "1530"] else "S_ECE.tif"
        ec_path = os.path.join(DATA_DIR, ec_file)
        if os.path.exists(ec_path):
            try:
                with rasterio.open(ec_path) as src:
                    ec_value = src.read(1)[row, col]
                    is_value = max(0.0446 * ec_value - 0.000173, 0)
                    params["IS"] = float(is_value)
            except:
                pass
        
        return params
        
    except Exception as e:
        return None

# 侧边栏
with st.sidebar:
    st.header("📊 参数设置")
    
    element = st.selectbox(
        "稀土元素",
        ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y"]
    )
    
    depth = st.selectbox(
        "土壤深度",
        ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"]
    )
    
    norm_method = st.selectbox(
        "归一化方法",
        ["原始数据", "百分位数归一化", "标准差归一化", "线性归一化"]
    )
    
    st.markdown("---")
    
    st.header("🔍 经纬度查询")
    col1, col2 = st.columns(2)
    with col1:
        lon = st.number_input("经度", min_value=73.0, max_value=135.0, value=105.0, step=0.1)
    with col2:
        lat = st.number_input("纬度", min_value=18.0, max_value=53.0, value=35.0, step=0.1)
    
    query_button = st.button("🎯 查询点位", use_container_width=True)
    
    st.markdown("---")
    show_debug = st.checkbox("显示调试信息", value=False)

# 深度映射
depth_mapping = {
    "0-5cm": "05", "5-15cm": "515", "15-30cm": "1530",
    "30-60cm": "3060", "60-100cm": "60100"
}
depth_suffix = depth_mapping[depth]

# 文件路径
raster_filename = f"prediction_result_{element}{depth_suffix}_raw.tif"
raster_path = os.path.join(DATA_DIR, raster_filename)

# 主界面
col_left, col_right = st.columns([2, 1])

with col_left:
    if not os.path.exists(raster_path):
        st.error(f"❌ 未找到文件: {raster_filename}")
        st.stop()
    
    # 加载数据
    data_info = load_raster_data(raster_path)
    
    if data_info is None:
        st.error("无法加载数据")
        st.stop()
    
    # 调试信息
    if show_debug:
        valid_data = data_info['data'].compressed() if np.ma.is_masked(data_info['data']) else data_info['data'].flatten()
        valid_data = valid_data[np.isfinite(valid_data)]
        if len(valid_data) > 0:
            st.info(f"""
            数据信息:
            - 形状: {data_info['data'].shape}
            - 范围: {np.min(valid_data):.4f} ~ {np.max(valid_data):.4f}
            - 平均值: {np.mean(valid_data):.4f}
            """)
    
    # 绘制地图
    try:
        # 数据处理
        display_data, vmin, vmax = normalize_data(data_info['data'], norm_method)
        
        # 创建图形（每次都创建新的）
        plt.close('all')
        fig = plt.figure(figsize=(12, 8))
        
        # 创建投影
        albers_proj = get_albers_projection()
        ax = plt.subplot(111, projection=albers_proj)
        
        # 地理要素
        ax.add_feature(cfeature.LAND, color='#f0f0f0', alpha=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='gray')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='gray')
        
        # 数据范围
        img_extent = (-2625683.87495, 2206316.12505, 1877102.875, 5921102.875)
        
        # 颜色映射
        cmap = 'viridis' if norm_method == "原始数据" else create_enhanced_colormap()
        
        # 绘制数据
        im = ax.imshow(
            display_data,
            origin='upper',
            extent=img_extent,
            transform=albers_proj,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.9,
            interpolation='nearest'
        )
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Kd值 [L/g]', fontsize=10)
        
        # 标题
        ax.set_title(f'{element}元素在{depth}土壤中的Kd值分布 ({norm_method})', fontsize=14)
        
        # 网格
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # 设置范围
        ax.set_extent(img_extent, crs=albers_proj)
        
        # 查询点标记
        if query_button:
            x, y = wgs84_to_albers(lon, lat, data_info['crs'])
            if x is not None and y is not None:
                ax.plot(x, y, 'ro', markersize=10, markeredgecolor='white', 
                       markeredgewidth=2, transform=albers_proj)
        
        # 显示图形
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        
    except Exception as e:
        st.error(f"地图绘制错误: {str(e)}")
        if show_debug:
            st.code(traceback.format_exc())

with col_right:
    st.subheader("📍 查询结果")
    
    if query_button:
        params = get_point_parameters(lon, lat, element, depth_suffix, data_info)
        
        if params:
            st.success("✅ 查询成功")
            st.write(f"**位置**: {lon:.4f}°E, {lat:.4f}°N")
            st.write(f"**元素**: {element} ({depth})")
            
            # 参数表格
            st.markdown("### 土壤参数")
            
            param_display = []
            param_info = {
                "Kd": ("L/g", "分配系数"),
                "pH": ("", "土壤酸碱度"),
                "SOM": ("g/kg", "有机质含量"),
                "CEC": ("cmol⁺/kg", "阳离子交换容量"),
                "IS": ("mol/L", "离子强度"),
                "Ce": ("mg/kg", "平衡浓度")
            }
            
            for param_name in ["Kd", "pH", "SOM", "CEC", "IS", "Ce"]:
                if param_name in params:
                    value = params[param_name]
                    unit, desc = param_info[param_name]
                    value_str = f"{value:.2f}" if value >= 1 else f"{value:.4f}"
                    param_display.append({
                        "参数": param_name,
                        "值": value_str,
                        "单位": unit
                    })
            
            df = pd.DataFrame(param_display)
            st.dataframe(df, hide_index=True, use_container_width=True)
            
        else:
            st.warning("⚠️ 该位置无有效数据")
    else:
        st.info('👆 点击"查询点位"按钮获取数据')
        
        # 空表格
        empty_df = pd.DataFrame({
            "参数": ["Kd", "pH", "SOM", "CEC", "IS", "Ce"],
            "值": ["--"] * 6,
            "单位": ["L/g", "", "g/kg", "cmol⁺/kg", "mol/L", "mg/kg"]
        })
        st.dataframe(empty_df, hide_index=True, use_container_width=True)

# 页脚
st.markdown("---")
st.markdown("🌱 稀土元素土壤Kd值可视化系统")