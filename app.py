import streamlit as st
import requests
from keras.models import load_model
import numpy as np
from PIL import Image
import cv2

# Function to download and load the model
def download_and_load_model(model_url):
    response = requests.get(model_url)
    model_path = "my_model.keras"
    with open(model_path, "wb") as f:
        f.write(response.content)
    model = load_model(model_path)
    return model

# URL of your model file
model_url = "https://drive.google.com/uc?id=1AUCvNE1a3Pp6PSjKE66KnCiCnSC1am1h"  # Change to direct download link
model = download_and_load_model(model_url)

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image for OCR
    image = image.convert('RGB')  # Ensure the image is in RGB format
    image = image.resize((128, 32))  # Resize to the expected input size of the model
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Perform OCR
    predictions = model.predict(image_array)
    
    # Decode predictions (Assuming your model outputs a sequence of class indices)
    # Implement your decoding logic based on your model's output
    # For example, you might have a character mapping
    def decode_predictions(preds):
        # Dummy decoding function, replace with your actual logic
        # This function should convert the model output to human-readable text
        return "Decoded text goes here"  # Replace with actual decoded output

    ocr_output = decode_predictions(predictions)

    # Display the OCR result
    st.write("OCR Output:")
    st.write(ocr_output)
