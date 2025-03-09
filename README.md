
# Fashion-Product-Classifier

## Overview
This project builds and deploys a **deep learning model** to predict multiple attributes of fashion products, including:
- **Article Type** (e.g., T-shirt, Jeans, Shoes)
- **Base Colour** (e.g., Red, Blue, Black)
- **Season** (e.g., Summer, Winter)
- **Gender** (e.g., Men, Women, Unisex)

The model is trained using **MobileNetV2** as a feature extractor and fine-tuned for multi-task classification. It is deployed as a **REST API** using Flask or FastAPI.

## Dataset
- **Source:** [Fashion Product Images Dataset - Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
- **Trained Model:** https://drive.google.com/file/d/14k1ifBlwHVOUI42voODcio30mnlRefS0/view?usp=sharing (Pickel File uploaded here)
- **Data:** Contains images of fashion products and a CSV file (`styles.csv`) with labels for product type, color, season, and gender.

## Features
- **Data Preprocessing:**  Exploratory Data Analysis (EDA),  Filtering Rows with Missing Images, Encoding Categorical Labels,Splitting the Dataset and image augmentation.
- **Multi-Task Deep Learning Model:** Uses **MobileNetV2** as a base model.
- **Model Training & Optimization:** Trained with **categorical cross-entropy loss** and **Adam optimizer**.
- **Deployment:** The trained model is saved (`.h5` format) and served via an API.
- **Dockerization:** Containerized using Docker for easy deployment.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fashion-product-prediction.git
   cd fashion-product-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and extract the dataset from [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset).

4. Train the model:
   ```bash
   fashion_product_attribute_prediction.ipynb
   ```

5. Run the API (FastAPI or Flask):
   ```bash
     app(flask).py (or) app(streamlit).py
    ```

## Model Training
- The **MobileNetV2** architecture is used with additional dense layers for multi-task classification.
- Data augmentation is applied using `ImageDataGenerator`.
- Model is trained for **10 epochs** using **batch size 32**.

## API Usage
Once deployed, you can send an image to the API endpoint `/predict`:

### Example Request (Using cURL)
```bash
curl -X POST "http://localhost:5000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@path/to/image.jpg"
```

### Example Response
```json
{
  "articleType": "Jeans",
  "baseColour": "Blue",
  "season": "Winter",
  "gender": "Men"
}
```

## Deployment with Docker
1. Build the Docker image:
   ```bash
   docker build -t fashion-predictor .
   ```
2. Run the container:
   ```bash
   docker run -p 5000:5000 fashion-predictor
   ```

## Results
- The model achieves **high accuracy** across all categories.
- It can predict multiple attributes from a **single image input**.

## Contributors
- **M Karthikeya Reddy** ([GitHub](https://github.com/karthikeyareddy183))

## License
This project is licensed under the MIT License. Feel free to contribute and improve it!

---
🔗 **GitHub Repo:** [(https://github.com/Karthikeyareddy183/Fashion-Product-Classifier)]

