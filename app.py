import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
import os

# Function to download model from a URL if necessary
def download_model(url, model_path):
    response = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(response.content)

# Path to the model
model_path = './my_model.keras'

# URL of the model (replace with your actual cloud storage URL if necessary)
model_url = 'https://drive.google.com/file/d/1AUCvNE1a3Pp6PSjKE66KnCiCnSC1am1h/view?usp=drive_link'

# Check if the model exists locally, if not, download it
if not os.path.exists(model_path):
    st.write("Downloading model...")
    download_model(model_url, model_path)
    st.write("Download complete!")

# Load the model
st.write("Loading model...")
model = load_model(model_path)
st.write("Model loaded successfully!")

# Function to preprocess image for model prediction
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to the input size of your model
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Image OCR Model")
st.write("Upload an image, and the model will process it.")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Predict with the model
    st.write("Processing the image...")
    predictions = model.predict(preprocessed_image)

    # Display prediction results (for example purposes, modify based on your actual output)
    st.write("Predictions:")
    st.write(predictions)

# Run the app locally with `streamlit run app.py`
