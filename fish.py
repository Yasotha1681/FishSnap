import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import pyrebase
import tensorflow as tf
import os

# Firebase configuration
firebaseConfig = {
    "apiKey": "AIzaSyAvWPPLyYWYoolAFboKTkvqqCNJNnYm01E",
    "authDomain": "fish-8f5de.firebaseapp.com",
    "databaseURL": "https://fish-8f5de-default-rtdb.firebaseio.com",
    "projectId": "fish-8f5de",
    "storageBucket": "fish-8f5de.appspot.com",
    "messagingSenderId": "860387876032",
    "appId": "1:860387876032:web:4ff9ec596963a67f76a125",
    "measurementId": "G-HQYHLKH6TT"
}

# Initialize Firebase
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Load your pre-trained model
model_path = 'path_to_your_model/fish_model.h5'  # Update this path
if os.path.isfile(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    st.error(f"Model file not found: {model_path}")

# Function to preprocess image for model prediction
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the size your model expects
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Placeholder function to identify fish
def identify_fish(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    class_names = ["Fish1", "Fish2", "Fish3"]  # Replace with your class names
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

# Signup function
def signup():
    st.sidebar.header("Sign Up")
    email = st.sidebar.text_input("Email", key="signup_email")
    password = st.sidebar.text_input("Password", type="password", key="signup_password")
    confirm_password = st.sidebar.text_input("Confirm Password", type="password", key="confirm_password")
    
    if st.sidebar.button("Sign Up"):
        if password == confirm_password:
            try:
                auth.create_user_with_email_and_password(email, password)
                st.sidebar.success("Account created successfully! You can now log in.")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
        else:
            st.sidebar.error("Passwords do not match!")

# Login function
def login():
    st.sidebar.header("Login")
    email = st.sidebar.text_input("Email", key="login_email")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    
    if st.sidebar.button("Login"):
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state['user'] = user
            st.sidebar.success("Logged in successfully!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# Logout function
def logout():
    if 'user' in st.session_state:
        st.sidebar.button("Logout", on_click=lambda: st.session_state.pop('user', None))
        st.sidebar.success("Logged out successfully!")

# Main application logic
if 'user' in st.session_state:
    st.title("Fish Identification App")
    
    # Sidebar
    with st.sidebar:
        st.header("Options")
        option = st.selectbox("Choose an option", ["Upload Image", "Visualize Data", "Show Progress"])
        logout()
    
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
                if 'model' in globals():
                    fish_name = identify_fish(image)
                    st.write(f"Identified Fish: {fish_name}")
                else:
                    st.error("Model not loaded. Check the model path.")
                
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

else:
    # Show signup and login options
    option = st.sidebar.selectbox("Select", ["Login", "Sign Up"])
    if option == "Sign Up":
        signup()
    elif option == "Login":
        login()




