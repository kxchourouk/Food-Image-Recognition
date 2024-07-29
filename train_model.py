import os
import base64
import numpy as np
from PIL import Image
from pymongo import MongoClient
from keras.utils import to_categorical
import tensorflow as tf

# Path to your image files
image_base_path = r"C:\Users\mochi\OneDrive\Desktop\Astrolab\Food-Image-Recognition-master\Data_Images"  # Ensure the path is correct

# MongoDB connection string
mongodb_uri = os.getenv('MONGODB_URI')

def get_image_data():
    client = MongoClient(mongodb_uri)
    db = client['Food_Recognition']
    collection = db['images']

    images = []
    labels = []

    for record in collection.find():
        filename = record.get("filename")
        category = record.get("category")
        
        if not filename or not category:
            continue
        
        image_path = os.path.join(image_base_path, filename)
        if not os.path.isfile(image_path):
            print(f"Image file not found: {image_path}")
            continue

        # Open and preprocess the image
        try:
            image = Image.open(image_path)
            image = image.resize((150, 150))  # Resize image to a fixed size
            image = np.array(image)
            if image.shape != (150, 150, 3):  # Check if image is correctly sized
                print(f"Unexpected image shape: {image.shape} for file: {filename}")
                continue
            
            images.append(image)
            labels.append(category)
        except Exception as e:
            print(f"Error processing image {filename}: {e}")

    if len(images) == 0 or len(labels) == 0:
        print("No images or labels found. Exiting.")
        exit()

    # Convert labels to numerical format
    label_set = sorted(set(labels))
    label_to_index = {label: index for index, label in enumerate(label_set)}
    y = np.array([label_to_index[label] for label in labels])
    y = to_categorical(y, num_classes=len(label_set))

    return np.array(images), y, len(label_set)

def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    images, labels, num_classes = get_image_data()
    if len(images) == 0 or len(labels) == 0:
        print("No images or labels found. Exiting.")
        exit()

    # Create and train the model
    model = create_model(input_shape=(150, 150, 3), num_classes=num_classes)
    model.fit(images, labels, epochs=10, batch_size=32, validation_split=0.2)
    
    # Save the model
    model.save('food_image_recognition_model.h5')
    print("Model trained and saved successfully.")