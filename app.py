import tensorflow
from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import os


# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['Food_Recognition']
images_and_categories_collection = db['images_and_categories']

# Define label meaning
labels = [
    'Shawarma', 'Cupcake', 'Majboos', 'Balaleet', 'Umm Ali', 'Tharid',
    'Warak Enab', 'Kunafa', 'Kousa Mahshi', 'Harees', 'Khubz', 'Luqaimat',
    'Pani Puri', 'Samboosa'
]

# Load the pre-trained model
tensorflow.keras.backend.clear_session()
model_best = load_model('best_model_101class.hdf5', compile=False)
print('Model successfully loaded!')

# Helper functions
def get_image_details(filename):
    return images_and_categories_collection.find_one({'filename': filename})

# Routes
@app.route('/')
def index():
    img = 'static\profile.jpg'
    return render_template('index.html', img=img)

@app.route('/recognize')
def magic():
    return render_template('recognize.html')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist("img")
    for f in files:
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return render_template('recognize.html')

@app.route('/predict')
def predict():
    result = []
    num_images = len(os.listdir(app.config['UPLOAD_FOLDER']))

    for i in range(num_images):
        filename = f'{i}.jpg'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        pred_img = image.load_img(file_path, target_size=(200, 200))
        pred_img = image.img_to_array(pred_img)
        pred_img = np.expand_dims(pred_img, axis=0) / 255.0

        pred = model_best.predict(pred_img)
        if np.isnan(pred).any():
            pred = np.array([0.05] * len(labels))  # Dummy values

        top_indices = pred[0].argsort()[-3:]
        top_labels = [labels[idx] for idx in top_indices]
        top_scores = [float("{:.2f}".format(pred[0][idx] * 100)) for idx in top_indices]

        # Fetch details from MongoDB
        details = []
        for label in top_labels:
            doc = images_and_categories_collection.find_one({'category': label})
            if doc:
                details.append({
                    'category': label,
                    'ingredients': doc.get('ingredients', []),
                    'calories': doc.get('calories', 0.0),
                    'score': top_scores[top_labels.index(label)]
                })

        result.append({
            'filename': file_path,
            'results': details
        })

    return render_template('results.html', results=result)

@app.route('/results')
def results():
    pack = []  # This will be populated with data from the database
    for item in images_and_categories_collection.find({}):  # Adjust query as needed
        pack.append({
            'image': item.get('image_url'),
            'result': item.get('recognition_results', {}),
            'food': item.get('nutrition_info_url', ''),
            'ingredients': ', '.join(item.get('ingredients', [])),
            'calories': item.get('calories', 0),
            'quantity': item.get('quantity', 100)  # Default quantity
        })
    
    whole_nutrition = []
    for item in images_and_categories_collection.find({}):  # Adjust query as needed
        whole_nutrition.append({
            'name': item.get('food_name', 'Unknown'),
            'value': item.get('calories', 0)  # or use any other relevant metric
        })
    
    return render_template('results.html', pack=pack, whole_nutrition=whole_nutrition)

@app.route('/update', methods=['POST'])
def update():
    return render_template('index.html', img='static/P2.jpg')

if __name__ == "__main__":
    app.run(debug=True)
