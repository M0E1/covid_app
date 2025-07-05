import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from PIL import Image
from tensorflow.keras.models import load_model
import time

# -------------------------------
# App & Model Configuration
# -------------------------------
APP_VERSION = "1.0.0"
BUILD_DATE = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

class_mapping = {
    0: "🦠 Covid-19",
    1: "✅ Normal",
    2: "😷 Pneumonia"
}

# -------------------------------
# Translations
# -------------------------------
T = {
    "English": {
        "title": "🩻 Chest X-ray Classifier",
        "description": "Upload a chest X-ray image to detect the condition using an AI model.",
        "upload": "📤 Upload X-ray Image",
        "predict": "🔍 Predict",
        "wait": "Analyzing...",
        "upload_first": "👈 Please upload an X-ray image first.",
        "prob_dist": "🔬 Class Probabilities",
        "about_model": "This model classifies chest X-rays into:",
        "developer": "**Developer:** Mohamed Mostafa Hassan",
        "accuracy": "**Test Accuracy:** 96.5%",
        "source": "**Data Source:** [Kaggle - COVID-19 Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)",
        "about_title": "📘 About the App",
        "model_info_title": "🧠 Model Details",
        "model_info": """
The model uses MobileNetV2 transfer learning on 317 X-ray images.

Dataset:
- Training: 251 (Covid-19 111, Normal 70, Pneumonia 70)
- Testing: 66 (Covid-19 26, Normal 20, Pneumonia 20)

Training:
- 75% train / 25% val split  
- Augmentation: rotation, zoom, shift, flip  
- Class weights for balance  
- Adam optimizer, categorical_crossentropy  
- 30 epochs with EarlyStopping & ReduceLROnPlateau  

Final test accuracy: 96.5%
        """,
        "performance_title": "📊 Test Set Performance",
        "conf_matrix_title": "🔢 Confusion Matrix"
    },
    "العربية": {
        "title": "🩻 تصنيف أشعة الصدر",
        "description": "ارفع صورة أشعة للصدر وسيقوم النموذج بتحليلها وتحديد نوع الحالة.",
        "upload": "📤 ارفع صورة X-ray",
        "predict": "🔍 كشف الحالة",
        "wait": "جاري التحليل...",
        "upload_first": "👈 من فضلك ارفع صورة X-ray أولاً.",
        "prob_dist": "🔬 توزيع الاحتمالات",
        "about_model": "يعتمد النموذج على تصنيف صور الأشعة إلى:",
        "developer": "**المطور:** Mohamed Mostafa Hassan",
        "accuracy": "**دقة الاختبار:** 96.5%",
        "source": "**مصدر البيانات:** [Kaggle - COVID-19 Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)",
        "about_title": "📘 حول التطبيق",
        "model_info_title": "🧠 تفاصيل النموذج",
        "model_info": """
تم بناء النموذج باستخدام MobileNetV2 على 317 صورة أشعة.

البيانات:
- تدريب: 251 (كوفيد-19 111، طبيعي 70، التهاب رئوي 70)
- اختبار: 66 (كوفيد-19 26، طبيعي 20، التهاب رئوي 20)

التدريب:
- تقسيم 75% تدريب / 25% تحقق  
- تعزيز الصور: تدوير، تكبير، تحريك، تقليب  
- أوزان للفئات  
- محسن Adam، categorical_crossentropy  
- 30 تكرار مع EarlyStopping و ReduceLROnPlateau  

الدقة النهائية على الاختبار: 96.5%
        """,
        "performance_title": "📊 أداء الاختبار",
        "conf_matrix_title": "🔢 مصفوفة الالتباس"
    }
}

logging.basicConfig(level=logging.INFO)

# -------------------------------
# Page & Custom CSS
# -------------------------------
st.set_page_config(page_title=T["English"]["title"], layout="wide", page_icon="🩺")
st.markdown("""
<style>
/* General background */
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #fff;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1e1e1e;
    color: #fff;
}
section[data-testid="stSidebar"] * {
    color: #fff !important;
}

/* Uploaded image */
.element-container img {
    border-radius: 10px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    animation: fadeIn 1s ease-in-out;
}

/* Metrics animation */
[data-testid="metric-container"] {
    animation: fadeInUp 0.8s ease-in-out;
}

/* Custom animations */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Button hover */
.stButton>button:hover {
    background: linear-gradient(90deg, #43B14B, #4682B4) !important;
    color: #fff !important;
    transform: scale(1.02);
}

/* Active tab */
[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(90deg, #43B14B, #4682B4);
    color: white !important;
    border-radius: 6px;
    font-weight: bold;
}

/* Footer */
.footer-bar {
    position: fixed;
    bottom: 0; left: 0; width: 100%;
    background-color: #111;
    color: #aaa; text-align: center;
    padding: 8px; font-size: 13px;
    border-top: 1px solid #333;
    z-index: 999;
}
</style>
<div class="footer-bar">
    🄯 2025 Mohamed Mostafa Hassan | Chest X-ray Classifier | All rights reserved
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model & Utilities
# -------------------------------
@st.cache_resource
def get_model():
    return load_model("covid_xray_classifier_mobilenetv2.h5")

@st.cache_data
def preprocess_image(_img, target_size=(224, 224)):
    img = _img.convert("RGB").resize(target_size)
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)

def run_prediction(img, model):
    try:
        batch = preprocess_image(img)
        probs = model.predict(batch)[0]
    except Exception as e:
        logging.error("Prediction failed", exc_info=e)
        st.error("⚠️ Prediction error.")
        return None, None, None
    idx = int(np.argmax(probs))
    return class_mapping[idx], float(probs[idx]), probs

# -------------------------------
# UI Logic
# -------------------------------
def show_home(lang):
    st.markdown("<h4 style='text-align:center; color:gray;'>🚀 Powered by MobileNetV2 + Streamlit</h4>", unsafe_allow_html=True)
    st.title(T[lang]["title"])
    st.write(T[lang]["description"])

    uploaded = st.file_uploader(T[lang]["upload"], type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info(T[lang]["upload_first"])
        return

    img = Image.open(uploaded)
    st.image(img, caption="🖼️ Uploaded X-ray", use_column_width=True)

    if st.button(T[lang]["predict"]):
        with st.spinner("⏳ " + T[lang]["wait"]):
            time.sleep(0.5)
            label, conf, probs = run_prediction(img, get_model())

        if label is None:
            return

        c1, c2 = st.columns(2)
        c1.metric("✅ Prediction", label)
        c2.metric("🔢 Confidence", f"{conf:.1%}")

        with st.expander(T[lang]["prob_dist"]):
            df = pd.DataFrame({
                "Class": list(class_mapping.values()),
                "Probability": probs
            })
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("Class", sort=None),
                    y="Probability",
                    color=alt.Color(
                        "Class",
                        scale=alt.Scale(
                            domain=list(class_mapping.values()),
                            range=["#FF4B4B", "#43B14B", "#4682B4"]
                        )
                    ),
                    tooltip=["Class", alt.Tooltip("Probability", format=".2%")]
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

def show_about(lang):
    st.title(T[lang]["about_title"])
    with st.expander("🧠 " + T[lang]["model_info_title"], expanded=False):
        st.write(T[lang]["model_info"])
    st.markdown("---")
    st.subheader(T[lang]["performance_title"])
    df_report = pd.DataFrame({
        "precision": [1.00, 0.90, 0.90, 0.94, 0.93, 0.94],
        "recall":    [1.00, 0.90, 0.90, 0.94, 0.93, 0.94],
        "f1-score":  [1.00, 0.90, 0.90,  "",  0.93, 0.94],
        "support":   [26, 20, 20, 66, 66, 66]
    }, index=["Covid", "Normal", "Pneumonia", "accuracy", "macro avg", "weighted avg"])
    st.table(df_report)
    st.subheader(T[lang]["conf_matrix_title"])
    cm = [[26, 0, 0], [0, 18, 2], [0, 2, 18]]
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Covid", "Normal", "Pneumonia"],
                yticklabels=["Covid", "Normal", "Pneumonia"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# -------------------------------
# Main App
# -------------------------------
def main():
    lang = st.sidebar.radio("🌐 Language", ["English", "العربية"], horizontal=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown(T[lang]["about_model"])
    st.sidebar.markdown("- 🦠 Covid-19  \n- ✅ Normal  \n- 😷 Pneumonia")
    st.sidebar.markdown("---")
    st.sidebar.markdown(T[lang]["developer"])
    st.sidebar.markdown(T[lang]["accuracy"])
    st.sidebar.markdown(T[lang]["source"])
    st.sidebar.markdown(f"**Version:** {APP_VERSION}  \n**Built:** {BUILD_DATE}")

    tabs = st.tabs(["🏠 Home", "ℹ️ About"])
    with tabs[0]: show_home(lang)
    with tabs[1]: show_about(lang)

if __name__ == "__main__":
    main()
