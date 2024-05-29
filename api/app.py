from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io

chest_model = load_model('/Users/sreehari/PycharmProjects/FInal_Project/models/chest.h5')
alz_model = load_model('/Users/sreehari/PycharmProjects/FInal_Project/models/alzheimer.h5')
arth_model = load_model('/Users/sreehari/PycharmProjects/FInal_Project/models/arthritis.h5')


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((128, 128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_chest(image_bytes):
    preprocessed_image = preprocess(image_bytes)
    predictions = chest_model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions[0])
    class_names = ['COVID-19', 'Pneumonia', 'Normal']
    return class_names[predicted_class], predictions[0][predicted_class]


def predict_alzheimers(image_bytes):
    preprocessed_image = preprocess(image_bytes)
    predictions = alz_model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions[0])
    class_names = ['Mild', 'Moderate', 'NonDementia']
    return class_names[predicted_class], predictions[0][predicted_class]

def predict_arthritis():
    image_bytes = request.files['image'].read()
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((150, 150))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    preprocessed_image = np.expand_dims(img_array, axis=0)
    predictions = arth_model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions[0])
    class_names = ['Normal', 'Arthritis']
    return class_names[predicted_class], predictions[0][predicted_class]



# Flask app
app = Flask(__name__)


@app.route('/chest', methods=['POST'])
def predict_chest_condition():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_bytes = request.files['image'].read()
    prediction, confidence = predict_chest(image_bytes)
    return jsonify(prediction)


@app.route('/alzheimer', methods=['POST'])
def predict_alzheimer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_bytes = request.files['image'].read()
    prediction, confidence = predict_alzheimers(image_bytes)
    return jsonify(prediction)


@app.route('/arthritis', methods=['POST'])
def predict_arthritis_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    prediction, confidence = predict_arthritis()
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
