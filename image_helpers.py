import os
from PIL import Image
import tensorflow as tf
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get MongoDB URI from environment variable
mongodb_uri = os.getenv('MONGODB_URI')

# MongoDB connection setup
client = MongoClient(mongodb_uri)
db = client['Food_Recognition']
collection = db['images_and_categories']

def download_image(filename, save_path):
    """Download image from MongoDB and save locally."""
    image_document = collection.find_one({'filename': filename})
    if image_document and 'image_data' in image_document:
        image_data = image_document['image_data']  # Assuming image_data contains binary data
        with open(save_path, 'wb') as f:
            f.write(image_data)
        return True
    return False

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image for TensorFlow model."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)  # Resize to the target size
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array / 255.0  # Normalize to [0, 1]
    return image_array

def create_dataset(image_paths, labels, batch_size=32):
    """Create a TensorFlow dataset from image paths and labels."""
    def load_image(image_path, label):
        image = load_and_preprocess_image(image_path)
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
