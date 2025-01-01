import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np

# Load Trained Model
def load_model():
    """
    Load the pre-trained TensorFlow model from the specified path.
    """
    model = tf.keras.models.load_model(r"C:\Users\Amina\Desktop\Enetcom Projects\IA_project\my_model.h5")
    return model

# Temporarily display a message while executing a block of code
with st.spinner('Model is being loaded...'):
    model = load_model()

# App title
st.write("""
         # 🩺 Skin Tumor Classification
         Upload an image of a skin tumor, and the model will classify it into one of 10 categories.
         """)

# File uploader widget
file = st.file_uploader("Please upload a Skin Tumor image (JPG or PNG format):", type=["jpg", "png"])

# Image preprocessing and prediction
def import_and_predict(image_data, model):
    """
    Preprocess the image and make predictions using the trained model.
    """
    size = (224, 224)  # Model's expected input size
    image = ImageOps.fit(image_data, size, Image.LANCZOS)  # Replaced ANTIALIAS with LANCZOS
    image = np.asarray(image)  # Convert image to numpy array
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image color to RGB
    img_reshape = img[np.newaxis, ...]  # Add batch dimension
    prediction = model.predict(img_reshape)
    return prediction

# Handle file upload and prediction
if file is None:
    st.text("Please upload an image file.")
else:
    # Display the uploaded image
    image = Image.open(file)
    st.image(image, use_container_width=True)  # Updated use_column_width to use_container_width
    
    # Make predictions
    predictions = import_and_predict(image, model)
    class_names = [
        "1. Eczema (1677)", "2. Melanoma (15.75k)", "3. Atopic Dermatitis (1.25k)", 
        "4. Basal Cell Carcinoma (BCC) (3323)", "5. Melanocytic Nevi (NV) (7970)", 
        "6. Benign Keratosis-like Lesions (BKL) (2624)", "7. Psoriasis (2k)", 
        "8. Seborrheic Keratoses (1.8k)", 
        "9. Tinea Ringworm (1.7k)", 
        "10. Warts Molluscum (2103)"
    ]
    
    # Display predictions
    predicted_class = class_names[np.argmax(predictions)]
    st.write(f"### 🧐 The skin tumor is classified as: **{predicted_class}**")
