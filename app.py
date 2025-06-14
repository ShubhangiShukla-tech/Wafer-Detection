
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Wafer Pass/Fail Detector", layout="centered")

model = tf.keras.models.load_model('wafer_pass_fail.h5')

st.markdown("<h1 style='text-align:center; color: #4B0082;'>🎯 Wafer Pass/Fail Detector</h1>", unsafe_allow_html=True)
st.markdown("<style>body {background-color: #F0F8FF;}</style>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload wafer image", type=["jpg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Wafer Image", use_column_width=True)
    img_resized = img.resize((128,128))
    data = np.array(img_resized)/255.0
    data = np.expand_dims(data,0)
    pred = model.predict(data)[0][0]
    result = "✅ PASS" if pred < 0.5 else "❌ FAIL"
    color = "#228B22" if pred < 0.5 else "#B22222"
    st.markdown(f"<h2 style='text-align:center; color:{color};'>{result}</h2>", unsafe_allow_html=True)
