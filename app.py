import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import requests

# Function to download the model from a cloud storage URL
def download_model(url, model_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise error for bad responses
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.write("Model downloaded successfully!")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download the model: {e}")

# Path to the model (local path after download)
model_path = './models/my_model.keras'

# Google Drive direct download URL for your model
model_url = 'https://drive.google.com/uc?export=download&id=1AUCvNE1a3Pp6PSjKE66KnCiCnSC1am1h'

# Download the model if it's not found locally
if not os.path.exists(model_path):
    st.write("Downloading model from cloud...")
    download_model(model_url, model_path)

# Load the model
if os.path.exists(model_path):
    st.write("Loading model...")
    model = load_model(model_path)
    st.write("Model loaded successfully!")
else:
    st.error(f"Model file not found at {model_path}. Please check the download link.")

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
