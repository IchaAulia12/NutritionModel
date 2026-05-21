import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ======================================
# CONFIG
# ======================================

IMG_SIZE = 224

st.set_page_config(
    page_title="AI Nutrition Scanner",
    page_icon="🍱",
    layout="centered"
)

# ======================================
# LOAD MODEL
# Rebuild arsitektur persis seperti di training,
# lalu load weights — agar tidak bergantung pada
# versi Keras saat model disimpan.
# ======================================

@st.cache_resource
def load_my_model():
    # --- Rebuild arsitektur (identik dengan notebook) ---
    base_model = MobileNetV2(
        weights=None,           # Tidak perlu imagenet, kita load dari .h5
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Build supaya layer ter-inisialisasi sebelum load weights
    model.build((None, IMG_SIZE, IMG_SIZE, 3))

    # --- Load weights dari file .h5 ---
    model.load_weights("best_nutrition_model.h5")

    return model

model = load_my_model()

# ======================================
# PREPROCESS IMAGE
# ======================================

def preprocess_image(image):
    img = image.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0          # Rescale sama seperti ImageDataGenerator
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ======================================
# PREDICT
# ======================================

def predict_nutrition(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    return {
        "Calories":     float(prediction[0]),
        "Fat":          float(prediction[1]),
        "Carbohydrate": float(prediction[2]),
        "Protein":      float(prediction[3]),
    }

# ======================================
# UI
# ======================================

st.title("🍱 AI Nutrition Scanner")
st.markdown("""
Upload gambar makanan dan AI akan memperkirakan:
**Calories · Fat · Carbohydrate · Protein**
""")

uploaded_file = st.file_uploader("Upload Food Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing nutrition..."):
        result = predict_nutrition(image)

    st.success("Prediction Completed!")

    # --- Metrics ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Calories", f"{result['Calories']:.2f} kcal")
        st.metric("Fat",      f"{result['Fat']:.2f} g")
    with col2:
        st.metric("Carbohydrate", f"{result['Carbohydrate']:.2f} g")
        st.metric("Protein",      f"{result['Protein']:.2f} g")

    # --- Table ---
    st.subheader("Nutrition Table")
    nutrition_df = pd.DataFrame({
        "Nutrient": ["Calories", "Fat", "Carbohydrate", "Protein"],
        "Value":    [result["Calories"], result["Fat"],
                     result["Carbohydrate"], result["Protein"]]
    })
    st.dataframe(nutrition_df, use_container_width=True)

    # --- Bar Chart ---
    st.subheader("Nutrition Visualization")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        nutrition_df["Nutrient"],
        nutrition_df["Value"],
        color=["#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF"]
    )
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_ylabel("Amount")
    ax.set_title("Predicted Nutrition Content")
    st.pyplot(fig)

    # --- Analysis ---
    st.subheader("AI Analysis")
    calories = result["Calories"]
    if calories < 200:
        st.info("🟢 This food appears to be **LOW calorie**.")
    elif calories < 500:
        st.warning("🟡 This food appears to be **MEDIUM calorie**.")
    else:
        st.error("🔴 This food appears to be **HIGH calorie**.")