from pymongo import MongoClient
import numpy as np
from PIL import Image
import os
import tensorflow as tf

mongodb_uri = os.getenv('MONGODB_URI')
# MongoDB connection setup
client = MongoClient(mongodb_uri)
db = client['Food_Recognition']
collection = db['images_and_categories']

def get_image_data():
    data = list(collection.find({}))
    image_paths = [item['filename'] for item in data]
    labels = [item['category'] for item in data]
    return image_paths, labels

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))  # Adjust size if needed
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return image

def create_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: (load_and_preprocess_image(x), y))
    dataset = dataset.shuffle(buffer_size=len(image_paths)).batch(32)
    return dataset

if __name__ == "__main__":
    image_paths, labels = get_image_data()
    dataset = create_dataset(image_paths, labels)
    # Save or use the dataset as needed
