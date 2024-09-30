import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import requests

# Function to download the model from a cloud storage URL
def download_model(url, model_path):
    response = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(response.content)

# Path to the model (local path after download)
model_path = 'my_model.keras'

# URL to the model (replace with your cloud storage URL)
model_url = 'https://drive.google.com/file/d/1AUCvNE1a3Pp6PSjKE66KnCiCnSC1am1h/view?usp=drive_link'

# Download the model if it's not found locally
if not os.path.exists(model_path):
    st.write("Downloading model from cloud...")
    download_model(model_url, model_path)
    st.write("Model downloaded successfully!")

# Load the model
st.write("Loading model...")
model = load_model(model_path)
st.write("Model loaded successfully!")

# Function to process the uploaded image
def process_image(image):
    img = Image.open(image)
    img = img.resize((128, 128))  # Resize based on model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch input
    return img_array

# Streamlit file uploader to ask for an image to process
st.title("Upload an image for OCR processing")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing the image...")

    # Preprocess the image and make predictions
    image_array = process_image(uploaded_file)
    prediction = model.predict(image_array)
    
    st.write(f"Prediction: {prediction}")
