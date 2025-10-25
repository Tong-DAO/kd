import streamlit as st
import numpy as np
import rasterio
from rasterio.warp import transform as coord_transform
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import pandas as pd
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ==================== 字体修复设置 ====================
# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加全局CSS字体支持
st.markdown("""
<style>
    /* 导入Google中文字体 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
    
    /* 全局字体设置 */
    * {
        font-family: 'Noto Sans SC', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans', sans-serif !important;
    }
    
    /* 确保所有Streamlit组件使用中文字体 */
    .stApp, .stSidebar, .stButton>button, .stSelectbox, .stNumberInput, .stTextInput, .stMarkdown {
        font-family: 'Noto Sans SC', 'Microsoft YaHei', 'SimHei', sans-serif !important;
    }
    
    /* 表格字体 */
    .dataframe {
        font-family: 'Noto Sans SC', 'Microsoft YaHei', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)
# ==================== 字体修复结束 ====================

# 页面配置
st.set_page_config(
    page_title="稀土元素土壤Kd值可视化",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题
st.title("🌱 稀土元素土壤Kd值可视化系统")

# 设置数据目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# 检查数据目录
if not os.path.exists(DATA_DIR):
    st.error(f"数据目录不存在: {DATA_DIR}")
    st.info("请将数据文件放置在以下目录: " + DATA_DIR)
    st.stop()

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
    data_copy = np.array(data, copy=True)
    
    if np.ma.is_masked(data):
        valid_data = data.compressed()
    else:
        valid_mask = np.isfinite(data_copy)
        valid_data = data_copy[valid_mask]
    
    if len(valid_data) == 0:
        return data_copy, 0, 1
    
    if method == "原始数据":
        data_copy[data_copy < 0] = 0
        return data_copy, 0, np.max(valid_data)
    
    elif method == "百分位数归一化":
        p5 = np.percentile(valid_data, 5)
        p95 = np.percentile(valid_data, 95)
        if p95 - p5 > 1e-10:
            normalized = (data_copy - p5) / (p95 - p5)
            normalized = np.clip(normalized, 0, 1)
            return normalized, 0, 1
        return data_copy, 0, 1
            
    elif method == "标准差归一化":
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        if std > 1e-10:
            normalized = (data_copy - mean) / (2 * std) + 0.5
            normalized = np.clip(normalized, 0, 1)
            return normalized, 0, 1
        return data_copy, 0, 1
            
    elif method == "线性归一化":
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        if max_val - min_val > 1e-10:
            normalized = (data_copy - min_val) / (max_val - min_val)
            return normalized, 0, 1
        return data_copy, 0, 1
    
    return data_copy, np.min(valid_data), np.max(valid_data)

@st.cache_data
def load_raster_data(file_path):
    """加载栅格数据"""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1).astype(np.float32)
            transform_matrix = src.transform
            crs = src.crs
            bounds = src.bounds
            
            # 处理无效值
            data[~np.isfinite(data)] = np.nan
            data = np.ma.masked_invalid(data)
            
            return {
                'data': data,
                'transform': transform_matrix,
                'crs': crs,
                'bounds': bounds
            }
    except Exception as e:
        return None

def get_point_parameters(lon, lat, element, depth_suffix, data_info):
    """获取指定点的所有参数值"""
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
        
        # 读取其他参数
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
                except:
                    pass
        
        # 计算IS
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
        
        return params, (row, col)
        
    except Exception:
        return None

def create_map_image(display_data, vmin, vmax, element, depth, norm_method, marker_point=None):
    """创建地图并返回图像字节流"""
    # 创建新图形
    fig = plt.figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # 选择颜色映射
    if norm_method == "原始数据":
        cmap = 'viridis'
    else:
        cmap = create_enhanced_colormap()
    
    # 显示数据
    im = ax.imshow(
        display_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
        interpolation='nearest'
    )
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Kd值 [L/g]', fontsize=10)
    
    # 设置标题 - 确保使用支持的字体
    ax.set_title(f'{element}元素在{depth}土壤中的Kd值分布 ({norm_method})', 
                 fontsize=12, fontfamily='DejaVu Sans')
    
    # 设置坐标轴
    ax.set_xlabel('列索引', fontsize=10, fontfamily='DejaVu Sans')
    ax.set_ylabel('行索引', fontsize=10, fontfamily='DejaVu Sans')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 添加查询点标记
    if marker_point is not None:
        row, col = marker_point
        ax.plot(col, row, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
        ax.annotate(
            '查询点',
            xy=(col, row),
            xytext=(col + 20, row - 20),
            fontsize=9,
            color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
        )
    
    # 调整布局
    plt.tight_layout()
    
    # 保存到字节流
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    # 关闭图形
    plt.close(fig)
    
    return buf

# 侧边栏
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
    
    query_button = st.button("🎯 查询点位", use_container_width=True, type="primary")
    
    st.markdown("---")
    show_stats = st.checkbox("显示统计信息", value=False)

# 深度映射
depth_mapping = {
    "0-5cm": "05", "5-15cm": "515", "15-30cm": "1530",
    "30-60cm": "3060", "60-100cm": "60100"
}
depth_suffix = depth_mapping[depth]

# 文件路径
raster_filename = f"prediction_result_{element}{depth_suffix}_raw.tif"
raster_path = os.path.join(DATA_DIR, raster_filename)

# 创建两列布局
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📊 Kd值空间分布图")
    
    if not os.path.exists(raster_path):
        st.error(f"❌ 未找到文件: {raster_filename}")
        st.info("请检查数据文件是否存在")
        st.stop()
    
    # 加载数据
    with st.spinner('正在加载数据...'):
        data_info = load_raster_data(raster_path)
    
    if data_info is None:
        st.error("无法加载数据文件")
        st.stop()
    
    # 显示统计信息
    if show_stats:
        valid_data = data_info['data'].compressed() if np.ma.is_masked(data_info['data']) else data_info['data'].flatten()
        valid_data = valid_data[np.isfinite(valid_data)]
        
        if len(valid_data) > 0:
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("最小值", f"{np.min(valid_data):.4f}")
            with col_stat2:
                st.metric("最大值", f"{np.max(valid_data):.4f}")
            with col_stat3:
                st.metric("平均值", f"{np.mean(valid_data):.4f}")
            with col_stat4:
                st.metric("中位数", f"{np.median(valid_data):.4f}")
    
    # 数据处理和显示
    try:
        # 归一化处理
        display_data, vmin, vmax = normalize_data(data_info['data'], norm_method)
        
        # 处理查询
        marker_point = None
        if query_button:
            result = get_point_parameters(lon, lat, element, depth_suffix, data_info)
            if result:
                params, marker_point = result
                st.session_state['query_result'] = params
            else:
                st.session_state['query_result'] = None
        
        # 生成地图图像
        with st.spinner('正在生成地图...'):
            img_buf = create_map_image(display_data, vmin, vmax, element, depth, norm_method, marker_point)
            
        # 显示图像 - 修复弃用参数
        st.image(img_buf, use_container_width=True)  # 修复：use_column_width -> use_container_width
        
    except Exception as e:
        st.error(f"地图生成错误: {str(e)}")

with col_right:
    st.subheader("📍 查询结果")
    
    # 检查是否有查询结果
    if 'query_result' in st.session_state and st.session_state['query_result'] is not None:
        params = st.session_state['query_result']
        
        st.success("✅ 查询成功")
        
        # 位置信息
        info_container = st.container()
        with info_container:
            st.markdown(f"""
            **📍 位置信息**
            - 经度: {lon:.4f}°E
            - 纬度: {lat:.4f}°N
            - 元素: {element}
            - 深度: {depth}
            """)
        
        st.markdown("---")
        
        # 参数表格
        st.markdown("**📊 土壤参数**")
        
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
        st.dataframe(df, hide_index=True, use_container_width=True)  # 修复：use_column_width -> use_container_width
        
        # 参数说明
        with st.expander("📖 参数说明"):
            st.markdown("""
            - **Kd**: 分配系数，表示元素在固液两相间的分配
            - **pH**: 土壤酸碱度
            - **SOM**: 土壤有机质含量
            - **CEC**: 阳离子交换容量
            - **IS**: 离子强度
            - **Ce**: 平衡浓度
            """)
    else:
        if query_button:
            st.warning("⚠️ 该位置无有效数据或超出范围")
        else:
            st.info("👆 请输入经纬度并点击查询按钮")
        
        # 显示空表格
        st.markdown("**📊 土壤参数**")
        empty_df = pd.DataFrame({
            "参数": ["Kd", "pH", "SOM", "CEC", "IS", "Ce"],
            "值": ["--"] * 6,
            "单位": ["L/g", "", "g/kg", "cmol⁺/kg", "mol/L", "mg/kg"]
        })
        st.dataframe(empty_df, hide_index=True, use_container_width=True)  # 修复：use_column_width -> use_container_width

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    🌱 稀土元素土壤Kd值可视化系统 v1.0<br>
    数据基于Albers等积圆锥投影 | 支持15种稀土元素分析
</div>
""", unsafe_allow_html=True)