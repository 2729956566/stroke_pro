import streamlit as st
import pandas as pd
import numpy as np
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from matplotlib import rcParams

# =============================================================================
# 页面基础配置
# =============================================================================

# 配置matplotlib以获得更好的可视化效果
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10

# 设置页面配置，包括页面标题、图标、布局和侧边栏初始状态
st.set_page_config(
    page_title="模型预测可视化",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 自定义CSS样式
# =============================================================================

# 使用st.markdown注入自定义CSS样式，美化界面
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #007bff; color: white; border-radius: 8px;}
    .stNumberInput>label {font-weight: bold; color: #2c3e50;}
    .sidebar .sidebar-content {background-color: #e9ecef;}
    h1 {color: #2c3e50; text-align: center;}
    h2 {color: #34495e; border-bottom: 2px solid #17a2b8; padding-bottom: 5px;}
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 标题和介绍
# =============================================================================

# 设置应用主标题
st.title("模型预测可视化")
# 应用介绍
st.markdown("""
    本工具利用特征数据进行预测，并通过SHAP可视化提供机理解释。
    在侧边栏调整特征值，观察预测结果和SHAP值的变化。
""")

# =============================================================================
# 数据与模型加载
# =============================================================================

# 使用缓存加载背景数据，提高性能
@st.cache_data
def load_background_data():
    """从Excel文件加载训练数据作为SHAP解释器的背景数据"""
    df = pd.read_excel('data/7feature_train.xlsx')
    return df.iloc[:, :-2]

# 使用缓存计算ROC曲线的最佳阈值
@st.cache_data
def calc_cut_off():
    """根据测试集和验证结果计算分类模型的最佳阈值"""
    df_y = pd.read_excel('data/7feature_test.xlsx').iloc[:, -1]
    df_pred_y = pd.read_excel('data/val_result.xlsx').iloc[:, -1]
    fpr, tpr, thresholds = roc_curve(df_y, df_pred_y)
    cut_off = thresholds[np.argmax(tpr - fpr)]
    return cut_off

# 使用缓存加载预训练的TensorFlow模型
@st.cache_resource
def load_model():
    """加载预训练的.h5模型文件"""
    return tf.keras.models.load_model('data/MODEL.h5')

# 初始化数据和模型
background_data = load_background_data()
model = load_model()
cut_off = calc_cut_off()

# 获取特征的默认值（使用背景数据的第一行）
default_values = background_data.iloc[0, :].to_dict()

# =============================================================================
# 侧边栏特征输入
# =============================================================================

# 侧边栏标题
st.sidebar.header("特征输入")
st.sidebar.markdown("请调整以下特征的数值：")

# 如果点击“重置为默认值”按钮，则将所有特征值恢复为默认值
if st.sidebar.button("重置为默认值", key="reset"):
    st.session_state.update(default_values)

# 获取所有特征的名称
features = list(default_values.keys())
print(features)

# 创建一个字典来存储用户输入的值
values = {}

# !!! 请在这里为您的特征添加单位 !!!
# 示例: '年龄': '(岁)', '身高': '(cm)'
# feature_units = {
#     'HE_TG': '甘油三酯 (mmol/L)',
#     'age': '年龄 (岁)',
#     'HE_HPfh1': '是否高血压（0：否，1：是）',
#     'BIA_PBF': '体脂率 (%)',
#     'BIA_BFM': '脂肪量 (kg)',
#     'HE_obe': '肥胖状态（1：体重过轻 2：正常 3：肥胖前期 4：1 期肥胖 5：2 期肥胖 6：3 期肥胖）',
#     'HE_BMI': 'BMI (kg/m²)'
# }

feature_units = {
    'HE_TG'    : '甘油三酯 (mmol/L)',
    'age'      : '年龄 (岁)',
    'HE_HPfh1' : '是否高血压（0：否，1：是）',
    'BIA_PBF'  : '体脂率 (%)',
    'BIA_BFM'  : '脂肪量 (kg)',
    'HE_obe'   : '肥胖状态（1：过轻 2：正常 3：肥胖前期 4-6：各期肥胖）',
    'HE_BMI'   : 'BMI (kg/m²)'
}


# 创建两列来布局输入框，使其更美观
cols = st.sidebar.columns(2)

# 遍历所有特征，为每个特征创建输入框
for i, feature in enumerate(features):
    # 将输入框分布在两列中
    with cols[i % 2]:
        # 获取特征的单位，如果未定义则为空字符串
        unit = feature_units.get(feature, "")
        label = f"{feature}\n {unit}"

        # 特别处理“是否高血压”特征
        if feature == 'HE_HFfh1':
            # 使用选择框代替数字输入框
            selected_option = st.selectbox(
                label,
                options=['否', '是'],
                key=feature
            )
            # 将用户的选择（“是”/“否”）映射为数值（1/0）
            values[feature] = 1 if selected_option == '是' else 0
        else:
            # 为其他特征创建数字输入框
            values[feature] = st.number_input(
                label,
                min_value=float(background_data[feature].min()),
                max_value=float(background_data[feature].max()),
                value=default_values[feature],
                step=0.001,
                format="%.3f",
                key=feature
            )

# =============================================================================
# 模型类型判断与主程序
# =============================================================================

@st.cache_data
def determine_model_type():
    """
    通过分析测试数据的目标变量（最后一列）来自动判断模型是分类还是回归。
    """
    try:
        # 读取目标变量
        df_y = pd.read_excel('data/7feature_test.xlsx').iloc[:, -1]
        
        # 判断唯一值的数量
        unique_values = df_y.nunique()
        
        # 如果唯一值小于等于2，或数据类型为对象/分类，则认为是分类模型
        if unique_values <= 2 or df_y.dtype in ['object', 'category']:
            model_type = "classification"
            # 获取类别标签
            labels = sorted(df_y.unique())
            return model_type, labels
        else:
            # 否则，认为是回归模型
            model_type = "regression"
            return model_type, None
            
    except Exception as e:
        st.error(f"在判断模型类型时发生错误: {str(e)}")
        return None, None

# 获取模型类型和类别标签
model_type, class_labels = determine_model_type()

# 主分析按钮
if st.button("开始分析计算", key="calculate"):
    # 将用户输入的特征值构造成DataFrame
    input_df = pd.DataFrame([values])
    
    # 进行模型预测
    prediction = model.predict(input_df.values, verbose=0)[0][0]
    
    # 使用容器显示预测结果
    with st.container():
        st.header("📈 预测结果")    
        col1, col2 = st.columns(2)
        
        with col1:
            # 根据模型类型显示不同的结果
            if model_type == "classification":
                # 分类模型：显示概率和预测类别
                predicted_class = class_labels[1] if prediction >= cut_off else class_labels[0]
                st.metric(
                    "预测概率", 
                    f"{prediction:.4f}", 
                    delta=f"预测类别: {predicted_class}",
                    delta_color="inverse"
                )
            else:
                # 回归模型：显示预测值
                st.metric(
                    "预测值", 
                    f"{prediction:.4f}"
                )
        
        with col2:
            if model_type == "classification":
                # 分类模型：显示分类阈值
                st.metric(
                    "分类阈值", 
                    f"{cut_off:.4f}"
                )
            else:
                # 回归模型：显示目标变量的范围作为参考
                df_y_for_range = pd.read_excel('data/7feature_test.xlsx').iloc[:, -1]
                st.metric(
                    "目标范围", 
                    f"{df_y_for_range.min():.2f} - {df_y_for_range.max():.2f}"
                )
    
    # =========================================================================
    # SHAP 可解释性分析
    # =========================================================================
    
    # 初始化SHAP解释器
    explainer = shap.DeepExplainer(model, background_data.values)
    # 计算SHAP值
    shap_values = np.squeeze(np.array(explainer.shap_values(input_df.values)))
    # 获取SHAP的基础值
    base_value = float(explainer.expected_value[0].numpy())

    # 使用选项卡展示不同的SHAP可视化图
    tab1, tab2, tab3 = st.tabs(["力图 (Force Plot)", "决策图 (Decision Plot)", "机理解释"])
    
    with tab1:
        st.subheader("力图 (Force Plot)")
        st.markdown("力图显示了每个特征是如何将预测结果从基线值“推动”到最终值的。红色特征增加预测值，蓝色特征降低预测值。")
        # 创建SHAP解释对象
        explanation = shap.Explanation(
            values=shap_values, 
            base_values=base_value, 
            feature_names=input_df.columns,
            data=input_df.values.round(3)
        )
        # 生成并显示力图
        shap.plots.force(explanation, matplotlib=True, show=False, figsize=(20, 4))
        st.pyplot(plt.gcf(), clear_figure=True)

    with tab2:
        st.subheader("决策图 (Decision Plot)")
        st.markdown("决策图展示了模型如何为单个样本做出决策。它从图的底部开始，显示了模型的基线值，然后每个特征的SHAP值被添加到模型输出中。")
        # 生成并显示决策图
        shap.decision_plot(base_value, shap_values, input_df.columns, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
    
    with tab3:
        st.subheader("机理解释")
        st.markdown("下表展示了每个特征对当前预测结果的具体贡献值（SHAP值）。正值表示该特征推动预测向高处发展，负值则相反。")
        # 创建一个DataFrame来显示特征及其SHAP值
        importance_df = pd.DataFrame({'特征': input_df.columns, 'SHAP Value': shap_values})
        # 按SHAP值降序排列
        importance_df = importance_df.sort_values('SHAP Value', ascending=False)
        # 使用颜色渐变来突出显示SHAP值的大小
        st.dataframe(importance_df.style.background_gradient(cmap='coolwarm', subset=['SHAP Value']))