import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

model = tf.keras.models.load_model("./RESNET_50.h5")
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.reshape(img_array, (1, 224, 224, 1))
    img_array = np.repeat(img_array, 3, axis=-1)
    return img_array

st.title("Covid-19 Detection App")
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)
    class_labels = ["COVID","NORMAL"]

    with col2:
        st.markdown("<div style='text-align:center;font-size:24px;font-weight:bold;'>Predicted Class:</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;font-size:36px;font-weight:bold;'>{class_labels[predicted_class[0]]}</div>", unsafe_allow_html=True)
