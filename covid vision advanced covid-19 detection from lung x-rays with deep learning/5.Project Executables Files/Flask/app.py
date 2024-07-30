from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
cnn = tf.keras.models.load_model('covid_cnn_model.h5')

# Helper function to load a single image
def load_single_image(file_path):
    img = Image.open(file_path).convert('RGB')
    img = img.resize((128, 128))  # Ensure the image is the same size
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/precautions')
def precautions():
    return render_template('precautions.html')

@app.route('/vaccination')
def vaccination():
    return render_template('vaccination.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        sample_image = load_single_image(file_path)
        prediction = cnn.predict(sample_image)
        class_label = np.argmax(prediction)
        result = "positive for COVID-19" if class_label == 0 else "negative for COVID-19"
        symptoms = ["Fever", "Dry cough", "Fatigue"] if class_label == 0 else []
        tips = ["Stay hydrated", "Rest", "Take fever reducers"] if class_label == 0 else []
        return render_template('result.html', result=result, symptoms=symptoms, tips=tips)

if __name__ == '__main__':
    app.run(debug=True)
