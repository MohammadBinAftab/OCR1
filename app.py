import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
import tempfile

# Function to download the model from a cloud storage URL directly
def download_and_load_model(url):
    try:
        # Create a temporary file to store the downloaded model
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as temp_file:
            model_path = temp_file.name
            
            # Download the model file from Google Drive
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad responses
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            st.write("Model downloaded successfully!")

            # Load the model from the temporary file
            model = load_model(model_path)
            st.write("Model loaded successfully!")
            return model
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download the model: {e}")
        return None

# Google Drive direct download URL for your model
model_url = 'https://drive.google.com/uc?export=download&id=1AUCvNE1a3Pp6PSjKE66KnCiCnSC1am1h'

# Directly download and load the model
st.write("Downloading and loading model from cloud...")
model = download_and_load_model(model_url)

if model is not None:
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
else:
    st.error("Model could not be loaded. Please check the download link.")
