import tensorflow as tf
import numpy as np
from PIL import Image
import os

class TensorFlowModel:
    def __init__(self):
        self.model_save_path = 'efficientnet_tf.h5'
        self.model = self.load_model()
        self.input_shape = (224, 224)  # Ensure this matches your model's expected input shape

    def load_model(self):
        if os.path.exists(self.model_save_path):
            print("Loading model...")
            model = tf.keras.models.load_model(self.model_save_path, compile=False)
            return model
        else:
            raise FileNotFoundError(f"Model file {self.model_save_path} not found")

    def preprocess_image(self, img):
        image = Image.open(img).convert('RGB')
        image = image.resize(self.input_shape)  # Resize to expected input shape
        img_array = np.array(image) / 255.0  # Normalize to [0, 1] range
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    def predict(self, img):
        img_array = self.preprocess_image(img)
        try:
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            return predicted_class
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
