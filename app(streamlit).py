
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load the model
model = load_model("fashion_product_model (1).h5")

# Load label encoders from the pickle file
try:
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load label encoders: {e}")
    label_encoders = {}

# Streamlit app
st.title("Fashion Product Classifier")
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get predictions from the model
    predictions = model.predict(img_array)

    # Convert predictions to categorical labels
    results = {}
    tasks = ['articleType', 'baseColour', 'season', 'gender']
    for i, task in enumerate(tasks):
        if task in label_encoders:
            predicted_class = np.argmax(predictions[i][0])
            results[task] = label_encoders[task].inverse_transform([predicted_class])[
                0]
        else:
            st.error(f"Label encoder for task '{task}' not found!")

    # Display the results
    st.write("Predicted Results:")
    st.write(results)
