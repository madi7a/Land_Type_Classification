import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("model.keras")

st.title("Land Use Classification (EuroSAT)")
st.write("Upload a satellite image (64x64 RGB) to predict land use.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Resize image to 224x224 to match model input size
    image = Image.open(uploaded_file).resize((224, 224))

    # Display uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    # Class names (adjust if you have a different list)
    class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
                   'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
                   'River', 'SeaLake']

    # Show prediction with confidence score
    st.write(f"**Prediction:** {class_names[class_index]} ({prediction[0][class_index]:.2f} confidence)")
