import pandas as pd
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
from pymongo import MongoClient
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Food Recognition App",
    page_icon="static/favicon.ico",  
    initial_sidebar_state="auto",
)

# MongoDB connection setup
mongodb_uri = os.getenv('MONGODB_URI')
client = MongoClient(mongodb_uri)
db = client['Food_Recognition']
collection = db['images_and_categories']

# Load pre-trained VGG16 model for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
model = Model(inputs=base_model.input, outputs=tf.keras.layers.Flatten()(base_model.output))

def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(img):
    img_array = preprocess_image(img)
    features = model.predict(img_array)
    return features.flatten()

def retrieve_all_image_paths():
    docs = collection.find()
    image_paths = []
    for doc in docs:
        filename = doc['filename']
        path = os.path.join('Data_Images', filename)
        if os.path.isfile(path):
            image_paths.append(path)
    return image_paths

def retrieve_all_categories():
    docs = collection.find()
    categories = []
    for doc in docs:
        categories.append(doc['category'])
    return categories

def find_most_similar_category(input_img):
    input_features = extract_features(input_img)
    
    image_paths = retrieve_all_image_paths()
    
    if not image_paths:
        st.error("No images found in the database.")
        return None, None
    
    similarities = []
    categories = []
    for path in image_paths:
        img = Image.open(path)
        img_features = extract_features(img)
        similarity = cosine_similarity([input_features], [img_features])
        similarities.append(similarity[0][0])
        doc = collection.find_one({'filename': os.path.basename(path)})
        if doc:
            categories.append(doc['category'])
    
    if not categories:
        st.error("No categories found.")
        return None, None
    
    most_similar_index = np.argmax(similarities)
    return categories[most_similar_index], similarities[most_similar_index]

def fetch_category_details(category):
    doc = collection.find_one({'category': category})
    if doc:
        return doc.get('ingredients', []), doc.get('calories', 0)
    else:
        return [], 0


# Existing Streamlit app content
st.title(":orange[Food Recognition App]")
with st.sidebar:
    st.image("restaurant.png", width=85)
    st.subheader("Identify your food and get nutritional details.")
    st.subheader("How It Works")
    st.markdown(
        """
        **1. Upload an image of your food.**  
        The app will recognize the food and provide nutritional information.

        **2. Get Nutritional Details:**  
        View the ingredients and calorie content of the recognized food.

        """
    )

st.divider()


# Upload file section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    
    predicted_category, _ = find_most_similar_category(img)
    
    # Map the predicted class to a category name (replace this with actual mapping)
    category_names = ["Shawarma", "Cupcake", "Majboos", "Balaleet", "Umm Ali", "Tharid", "Warak Enab", "Kunafa", "Kousa Mahshi", "Harees", "Khubz", "Luqaimat", "Pani Puri", "Samboosa"]    
    
    st.write(f"Predicted Name: {predicted_category}")

    ingredients,calories=fetch_category_details(predicted_category)
    # Split ingredients by comma, then join with newline
    if isinstance(ingredients, str):
        ingredients = ingredients.split(',')
    ingredients_formatted = '\n'.join(f"- {ingredient.strip()}" for ingredient in ingredients)
    
    # Remove decimal places from calories
    if calories != 'Unknown':
        calories = int(float(calories))
    
    # Create a DataFrame for display
    data = {
        "Ingredients": [ingredients_formatted],
        "Calories": [calories]
    }
    df = pd.DataFrame(data, index=[predicted_category])

    # Display the result in a table with custom formatting
    st.write("### Food Details")
    st.table(df.style.set_properties(subset=['Ingredients'], **{'white-space': 'pre-wrap'}))


# Go back to top
st.markdown("[:arrow_up: Back to Top](#food-recognition-app)")
