from pymongo import MongoClient
from PIL import Image
import io
import base64
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import os

mongodb_uri = os.getenv('MONGODB_URI')
# MongoDB connection setup
client = MongoClient(mongodb_uri)
db = client['Food_Recognition']
collection = db['images_and_categories']

def get_image_data():
    images, labels = [], []
    for record in collection.find():
        image_data = base64.b64decode(record["image"])
        image = Image.open(io.BytesIO(image_data))
        images.append(image)
        labels.append(record["category"])
    return images, labels

def preprocess_images(images):
    processed_images = []
    for image in images:
        image = image.resize((224, 224))  # Adjust size as needed
        image = img_to_array(image) / 255.0  # Normalize to [0, 1]
        processed_images.append(image)
    return np.array(processed_images)

def load_and_preprocess_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)
