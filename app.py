import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load your OCR model
model = load_model('my_model.keras')  # Update the path for local use

# Title of the app
st.title("OCR Image Processing App")

# Ask for an image to upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Open the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for your model (update as per your model's input requirement)
    image = image.resize((128, 128))  # Example resizing, adjust as needed
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make predictions
    st.write("Processing...")
    prediction = model.predict(image)
    
    # Display results (adjust as per your model output)
    st.write(f"Prediction: {prediction}")
