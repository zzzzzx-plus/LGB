import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import math
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'MLPmodel.pkl')
with open(model_path, 'rb') as file:
    model = joblib.load(file)  # 使用 pickle 加载模型文件

# 特征名称
feature_names = [
    "Age", "Symptom Duration", "Flexion", "Extension", "Rotation",
    "Spring test-pain", "Muscle tightness", "Exacerbation on Flexion",
    "Exacerbation on Extension"
]

# Streamlit 应用标题
st.title("A Quick Screening Tool for Predicting NP Patients Benefiting from SMT")   

# 输入特征
age = st.number_input("Age:", min_value=12, max_value=72, value=50, step=1)
symptom_duration = st.number_input("Symptom Duration(days):", min_value=1, max_value=360, value=25, step=1)
flexion = st.number_input("Flexion:", min_value=15, max_value=60, value=40, step=1)
extension = st.number_input("Extension:", min_value=15, max_value=50, value=36, step=1)
rotation_rom = st.number_input("Rotation:", min_value=35, max_value=80, value=50, step=1)
spring_test_pain = st.selectbox("Spring test-pain (0=No, 1=mild, 2=middle, 3=severe):", options=[0, 1, 2, 3])
muscle_tightness = st.selectbox("Muscle tightness (0=No, 1=mild, 2=middle, 3=severe):", options=[0, 1, 2, 3])
exacerbation_on_flexion = st.selectbox("Exacerbation on Flexion (0=No, 1=Yes):", options=[0, 1])
exacerbation_on_extension = st.selectbox("Exacerbation on Extension (0=No, 1=Yes):", options=[0, 1])

# 整合特征
feature_values = [
    age, symptom_duration, flexion, extension, rotation_rom,
    spring_test_pain, muscle_tightness, exacerbation_on_flexion, exacerbation_on_extension,
]
features = pd.DataFrame([feature_values], columns=feature_names)

# 预测按钮

if st.button("Prediction"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to the model, your probability of benefiting from spinal manipulation is {probability:.1f}%."
        )
    else:
        advice = (
            f"According to the model, your probability of not benefiting from spinal manipulation is {probability:.1f}%."
        )

    # 调整字体大小
    font_size = "18px"  # 可根据需要设置字体大小
    advice_html = f"""
    <div style="font-size: {font_size};">
        {advice}
    </div>
    """
    st.markdown(advice_html, unsafe_allow_html=True)

    # SHAP 分析
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # 获取 SHAP 值，并分别计算正负两边贡献值最大的特征
    shap_values_single = shap_values[0]  # 当前样本的 SHAP 值
    top_positive_idx = tuple(shap_values_single.argsort()[-3:][::-1]) # 贡献值最大的 3 个正向特征
    top_negative_idx = tuple(shap_values_single.argsort()[:3])  # 贡献值最大的 3 个负向特征

    # 构造特征名称列表，只保留正负两边前两名特征，其余为空字符串
    top_features = set(top_positive_idx).union(set(top_negative_idx))
    display_feature_names = [
        feature_names[i] if i in top_features else "" for i in range(len(feature_names))
    ]

    # 生成并显示静态 SHAP 力图
    shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values_single,
        features=display_feature_names,  # 替换为正负两边前两名特征名称
        feature_names=None,  # 不显示具体特征值
        matplotlib=True
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
    st.image("shap_force_plot.png")
    
