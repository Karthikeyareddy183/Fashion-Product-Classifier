from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
import joblib

# Initialize Flask app
app = Flask(__name__)

# Check if the model file exists
if not os.path.exists("fashion_product_model (1).h5"):
    raise FileNotFoundError("The 'fashion_product_model (1).h5' file was not found. Please ensure it is in the correct directory.")

# Load the trained model
model = load_model("fashion_product_model (1).h5")

# Load the label encoders
if not os.path.exists("label_encoders.pkl"):
    raise FileNotFoundError("The 'label_encoders.pkl' file was not found. Please ensure it is in the correct directory.")
label_encoders = joblib.load("label_encoders.pkl")

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join("uploads", filename)
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Make predictions
        predictions = model.predict(img_array)

        # Decode the predictions using the label encoders
        results = {
            "articleType": label_encoders['articleType'].inverse_transform([np.argmax(predictions[0])])[0],
            "baseColour": label_encoders['baseColour'].inverse_transform([np.argmax(predictions[1])])[0],
            "season": label_encoders['season'].inverse_transform([np.argmax(predictions[2])])[0],
            "gender": label_encoders['gender'].inverse_transform([np.argmax(predictions[3])])[0]
        }

        # Clean up: Delete the uploaded file
        os.remove(file_path)

        return jsonify(results)

    return jsonify({"error": "File type not allowed"}), 400

# Run the Flask app
if __name__ == '__main__':
    # Create the uploads directory if it doesn't exist
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
