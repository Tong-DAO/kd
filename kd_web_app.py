import streamlit as st
import numpy as np
import rasterio
from rasterio.warp import transform
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import traceback

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

# 状态初始化
if 'map_placeholder' not in st.session_state:
    st.session_state.map_placeholder = None

# 应用标题
st.title("🌱 稀土元素土壤Kd值可视化")

# 设置数据目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# 检查数据目录
if not os.path.exists(DATA_DIR):
    st.error(f"数据目录不存在: {DATA_DIR}")
    st.info("请将数据文件放置在以下目录: " + DATA_DIR)
    st.stop()

# 初始化session state用于数据缓存
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

# 定义Albers投影
@st.cache_data
def get_albers_projection():
    """创建与数据匹配的Albers投影"""
    return ccrs.AlbersEqualArea(
        central_longitude=105,
        standard_parallels=(25, 47),
        false_easting=0,
        false_northing=0,
        globe=ccrs.Globe(datum="WGS84")
    )

def wgs84_to_albers(lon, lat, crs):
    """将经纬度转换为Albers坐标"""
    x, y = transform('EPSG:4326', crs, [lon], [lat])
    return x[0], y[0]

def create_enhanced_colormap():
    """创建增强对比度的颜色映射"""
    colors = ['#00008B', '#0000FF', '#0080FF', '#00BFFF',
             '#00FF80', '#80FF00', '#FFFF00', '#FF8000',
             '#FF0000', '#8B0000']
    return LinearSegmentedColormap.from_list('enhanced_viridis', colors, N=256)

def normalize_data(data, method):
    """对数据进行归一化处理"""
    valid_data = data[~np.ma.getmask(data)] if np.ma.is_masked(data) else data
    
    if len(valid_data) == 0:
        return data, 0, 1
    
    if method == "原始数据":
        # 原始数据，将负值设为0
        processed_data = np.where(data < 0, 0, data)
        if np.ma.is_masked(data):
            processed_data = np.ma.masked_array(processed_data, mask=data.mask)
        return processed_data, 0, np.max(valid_data)
    
    elif method == "百分位数归一化":
        p5 = np.percentile(valid_data, 5)
        p95 = np.percentile(valid_data, 95)
        if p95 - p5 > 0:
            normalized = (data - p5) / (p95 - p5)
            normalized = np.clip(normalized, 0, 1)
            return normalized, 0, 1
            
    elif method == "标准差归一化":
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        if std > 0:
            normalized = (data - mean) / (2 * std) + 0.5
            normalized = np.clip(normalized, 0, 1)
            return normalized, 0, 1
            
    elif method == "线性归一化":
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        if max_val - min_val > 0:
            normalized = (data - min_val) / (max_val - min_val)
            return normalized, 0, 1
    
    return data, np.min(valid_data), np.max(valid_data)

def load_raster_data(file_path, cache_key):
    """加载栅格数据并缓存"""
    if cache_key in st.session_state.data_cache:
        return st.session_state.data_cache[cache_key]
    
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1).astype(np.float32)
            transform_matrix = src.transform
            crs = src.crs
            nodata = src.nodata
            
            # 处理NoData
            data = np.ma.masked_invalid(data)
            
            result = {
                'data': data,
                'transform': transform_matrix,
                'crs': crs,
                'nodata': nodata
            }
            
            st.session_state.data_cache[cache_key] = result
            return result
            
    except Exception as e:
        st.error(f"加载文件失败: {file_path}")
        st.error(str(e))
        return None

def get_point_parameters(lon, lat, element, depth_suffix, data_info):
    """获取指定点的所有参数值"""
    try:
        # 转换坐标
        x, y = wgs84_to_albers(lon, lat, data_info['crs'])
        
        # 计算栅格索引
        row, col = rasterio.transform.rowcol(data_info['transform'], x, y)
        
        # 检查范围
        if (row < 0 or row >= data_info['data'].shape[0] or 
            col < 0 or col >= data_info['data'].shape[1]):
            return None
        
        # 获取Kd值
        kd_value = data_info['data'][row, col]
        if np.ma.is_masked(kd_value):
            return None
        
        params = {"Kd": float(kd_value)}
        
        # 读取pH值
        ph_file = os.path.join(DATA_DIR, f"ph{depth_suffix}.tif")
        if os.path.exists(ph_file):
            with rasterio.open(ph_file) as src:
                ph_value = src.read(1)[row, col] / 100
                params["pH"] = float(ph_value)
        
        # 读取SOM值
        som_file = os.path.join(DATA_DIR, f"soc{depth_suffix}.tif")
        if os.path.exists(som_file):
            with rasterio.open(som_file) as src:
                som_value = src.read(1)[row, col] * 1.724 / 100
                params["SOM"] = float(som_value)
        
        # 读取CEC值
        cec_file = os.path.join(DATA_DIR, f"cec{depth_suffix}.tif")
        if os.path.exists(cec_file):
            with rasterio.open(cec_file) as src:
                cec_value = src.read(1)[row, col] / 100
                params["CEC"] = float(cec_value)
        
        # 读取EC值并计算IS
        if depth_suffix in ["05", "515", "1530"]:
            ec_file = os.path.join(DATA_DIR, "T_ECE.tif")
        else:
            ec_file = os.path.join(DATA_DIR, "S_ECE.tif")
            
        if os.path.exists(ec_file):
            with rasterio.open(ec_file) as src:
                ec_value = src.read(1)[row, col]
                is_value = max(0.0446 * ec_value - 0.000173, 0)
                params["IS"] = float(is_value)
        
        # 读取Ce值
        ce_file = os.path.join(DATA_DIR, f"{element}.tif")
        if os.path.exists(ce_file):
            with rasterio.open(ce_file) as src:
                ce_value = src.read(1)[row, col]
                params["Ce"] = float(ce_value)
        
        return params
        
    except Exception as e:
        st.error(f"获取参数时出错: {str(e)}")
        return None

# 侧边栏参数设置
with st.sidebar:
    st.header("📊 参数设置")
    
    element = st.selectbox(
        "稀土元素",
        ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y"],
        help="选择要显示的稀土元素"
    )
    
    depth = st.selectbox(
        "土壤深度",
        ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"],
        help="选择土壤采样深度"
    )
    
    norm_method = st.selectbox(
        "归一化方法",
        ["原始数据", "百分位数归一化", "标准差归一化", "线性归一化"],
        help="选择数据归一化方法"
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
    
    # 显示调试信息选项
    show_debug = st.checkbox("显示调试信息", value=False)

# 深度映射
depth_mapping = {
    "0-5cm": "05", "5-15cm": "515", "15-30cm": "1530",
    "30-60cm": "3060", "60-100cm": "60100"
}
depth_suffix = depth_mapping[depth]

# 构建文件路径
raster_filename = f"prediction_result_{element}{depth_suffix}_raw.tif"
raster_path = os.path.join(DATA_DIR, raster_filename)

# 主界面布局
col_left, col_right = st.columns([2, 1])

with col_left:
    # 检查文件存在
    if not os.path.exists(raster_path):
        st.error(f"❌ 未找到文件: {raster_filename}")
        st.info("请检查数据文件是否存在于data目录中")
        st.stop()
    
    # 加载数据
    cache_key = f"{element}_{depth_suffix}"
    data_info = load_raster_data(raster_path, cache_key)
    
    if data_info is None:
        st.error("无法加载数据")
        st.stop()
    
    # 显示调试信息
    if show_debug:
        st.info(f"数据形状: {data_info['data'].shape}")
        valid_data = data_info['data'][~data_info['data'].mask] if np.ma.is_masked(data_info['data']) else data_info['data']
        if len(valid_data) > 0:
            st.info(f"数值范围: {np.min(valid_data):.4f} - {np.max(valid_data):.4f}")
            st.info(f"平均值: {np.mean(valid_data):.4f}, 中位数: {np.median(valid_data):.4f}")
    
    # 创建地图
    try:
        # 数据归一化
        display_data, vmin, vmax = normalize_data(data_info['data'], norm_method)
        
        # 获取数据范围（Albers投影）
        left = -2625683.87495
        right = 2206316.12505
        bottom = 1877102.875
        top = 5921102.875
        img_extent = (left, right, bottom, top)
        
        # 创建图形
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 1, height_ratios=[29, 1], hspace=0.15)
        
        # 创建地图子图
        albers_proj = get_albers_projection()
        ax = fig.add_subplot(gs[0], projection=albers_proj)
        
        # 添加地理要素
        ax.add_feature(cfeature.LAND, color='#f0f0f0', zorder=1)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3, zorder=1)
        ax.add_feature(cfeature.BORDERS, edgecolor='gray', linewidth=0.8, zorder=3)
        ax.add_feature(cfeature.COASTLINE, edgecolor='gray', linewidth=0.8, zorder=3)
        
        # 选择颜色映射
        if norm_method == "原始数据":
            cmap = 'viridis'
        else:
            cmap = create_enhanced_colormap()
        
        # 绘制栅格数据
        im = ax.imshow(
            display_data,
            origin='upper',
            extent=img_extent,
            transform=albers_proj,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.9,
            interpolation='nearest',
            zorder=2
        )
        
        # 添加颜色条
        cax = fig.add_subplot(gs[1])
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal', pad=0.05)
        cbar.set_label('Kd值 [L/g]', fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        
        # 设置标题
        ax.set_title(f'{element}元素在{depth}土壤中的Kd值分布 ({norm_method})', fontsize=14, pad=10)
        
        # 添加网格
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color='gray',
            alpha=0.5,
            linestyle='--'
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        
        # 设置显示范围
        ax.set_extent(img_extent, crs=albers_proj)
        
        # 如果有查询，添加标记点
        if query_button:
            x, y = wgs84_to_albers(lon, lat, data_info['crs'])
            ax.plot(x, y, 'ro', markersize=10, markeredgecolor='white', 
                   markeredgewidth=2, transform=albers_proj, zorder=5)
        
        # 使用占位符管理地图组件
        if st.session_state.map_placeholder is None:
            st.session_state.map_placeholder = st.empty()
        
        with st.session_state.map_placeholder.container():
            st.pyplot(fig, use_container_width=True)
        
        # 关闭图形释放内存
        plt.close(fig)
        
    except Exception as e:
        st.error(f"地图绘制错误: {str(e)}")
        if show_debug:
            st.code(traceback.format_exc())

with col_right:
    st.subheader("📍 查询结果")
    
    # 查询结果显示
    if query_button:
        # 重置地图占位符
        if st.session_state.map_placeholder is not None:
            st.session_state.map_placeholder.empty()
            st.session_state.map_placeholder = None
        
        params = get_point_parameters(lon, lat, element, depth_suffix, data_info)
        
        if params:
            st.success(f"✅ 查询成功")
            st.write(f"**位置**: {lon:.4f}°E, {lat:.4f}°N")
            st.write(f"**元素**: {element} ({depth})")
            
            # 显示参数表格
            st.markdown("### 土壤参数")
            
            # 格式化显示参数
            param_display = {
                "参数": [],
                "值": [],
                "单位": []
            }
            
            param_units = {
                "Kd": "L/g",
                "pH": "",
                "SOM": "g/kg",
                "CEC": "cmol⁺/kg",
                "IS": "mol/L",
                "Ce": "mg/kg"
            }
            
            for param_name in ["Kd", "pH", "SOM", "CEC", "IS", "Ce"]:
                if param_name in params:
                    param_display["参数"].append(param_name)
                    value = params[param_name]
                    if value >= 1:
                        param_display["值"].append(f"{value:.2f}")
                    else:
                        param_display["值"].append(f"{value:.4f}")
                    param_display["单位"].append(param_units[param_name])
            
            import pandas as pd
            df = pd.DataFrame(param_display)
            st.dataframe(df, hide_index=True, use_container_width=True)
            
            # 添加数据解释
            with st.expander("参数说明"):
                st.markdown("""
                - **Kd**: 分配系数，表示元素在固液两相间的分配
                - **pH**: 土壤酸碱度
                - **SOM**: 土壤有机质含量
                - **CEC**: 阳离子交换容量
                - **IS**: 离子强度
                - **Ce**: 平衡浓度
                """)
        else:
            st.warning("⚠️ 该位置无有效数据或超出数据范围")
    else:
        st.info("👆 点击'查询点位'按钮获取数据")
        
        # 显示空表格
        st.markdown("### 土壤参数")
        empty_data = {
            "参数": ["Kd", "pH", "SOM", "CEC", "IS", "Ce"],
            "值": ["--"] * 6,
            "单位": ["L/g", "", "g/kg", "cmol⁺/kg", "mol/L", "mg/kg"]
        }
        import pandas as pd
        df = pd.DataFrame(empty_data)
        st.dataframe(df, hide_index=True, use_container_width=True)

# 页脚信息
st.markdown("---")
st.markdown("🌱 稀土元素土壤Kd值可视化系统 | 数据基于Albers等积圆锥投影")