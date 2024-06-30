# app.py

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model.keras')

# Define class names
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


# Preprocess the uploaded image
def preprocess_image(image):
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


st.title("Image Classification with CNN")
st.write("Upload an image and the model will predict the category.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_array = preprocess_image(image)

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

