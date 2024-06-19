# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
@st.cache
def get_model():
    return load_model("Traffic_sign_classifier_CNN.hdf5")

model = get_model()

# Function to preprocess the input image
def preprocess_image(image):
    # Resize to 32x32 pixels (same size as the training data)
    resized_image = cv2.resize(image, (32, 32))
    # Normalize pixel values
    normalized_image = resized_image / 255.0
    return normalized_image

# Function to predict the traffic sign
def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    predicted_class = np.argmax(prediction)
    return predicted_class

# Streamlit app
def main():
    st.title("Traffic Sign Classifier")
    st.write("Upload an image of a traffic sign to classify it!")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = np.array(uploaded_image.read())
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        predicted_class = predict(image)

        # Load class names
        class_names = pd.read_csv("signnames.csv")
        sign_name = class_names.iloc[predicted_class]["SignName"]

        st.success(f"Predicted Traffic Sign: {sign_name}")

if __name__ == "__main__":
    main()
