import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import time

# Load your CNN model
model_path = 'C:/Users/SSNiTHAR/Desktop/ITBIN-2110-0126/test_model.h5'

if os.path.isfile(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
else:
    st.error(f"Model file not found: {model_path}")

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the size your model expects
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def identify_fish(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    class_names = ["Fish1", "Fish2", "Fish3"]  # Replace with your class names
    
    # Ensure the number of class names matches the model output
    if len(predictions[0]) != len(class_names):
        st.error("Model output size does not match the number of class names.")
        return "Unknown"

    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

# Main application logic
st.title("Fish Identification App")

# Sidebar
with st.sidebar:
    st.header("Options")
    option = st.selectbox("Choose an option", ["Upload Image", "Visualize Data", "Show Progress"])

# Containers
container1 = st.container()

if option == "Upload Image":
    with container1:
        st.header("Upload Fish Image")
        uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Fish Image', use_column_width=True)

            # Identify the fish
            fish_name = identify_fish(image)
            st.write(f"Identified Fish: {fish_name}")

            st.success("Image uploaded and identified successfully!")

elif option == "Visualize Data":
    with container1:
        st.header("Data Visualization")
        st.subheader("Random Data")
        data = np.random.randn(100)
        fig, ax = plt.subplots()
        ax.hist(data, bins=20)
        st.pyplot(fig)

elif option == "Show Progress":
    with container1:
        st.header("Progress and Status")
        st.subheader("Progress Bar")
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar.progress(i + 1)
        st.success("Task completed!")

        st.subheader("Status Messages")
        st.info("Information message")
        st.warning("Warning message")
        st.error("Error message")


    







    





