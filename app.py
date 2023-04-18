from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.utils import img_to_array

# Load the trained model
model = tf.keras.models.load_model('trained_model.h5')  

# Create Flask app
app = Flask(__name__)

# Define class labels mapping
# Define a mapping of folder names to labels
label_mapping = {
    'Butterfly': 0,
    'Moths and butterflies': 1,
    'Lizard': 2,
    'Spider': 3,
    'Fish': 4,
    'Monkey': 5,
    'Eagle': 6,
    'Frog': 7,
    'Snake': 8,
    'Duck': 9,
    'Caterpillar': 10,
    'Sparrow': 11,
    'Jellyfish': 12,
    'Parrot': 13,
    'Owl': 14,
    'Horse': 15,
    'Ladybug': 16,
    'Tortoise': 17,
    'Chicken': 18,
    'Penguin': 19,
    'Snail': 20,
    'Squirrel': 21,
    'Deer': 22,
    'Tiger': 23,
    'Crab': 24,
    'Shark': 25,
    'Giraffe': 26,
    'Goose': 27,
    'Whale': 28,
    'Starfish': 29,
    'Harbor seal': 30,
    'Sea turtle': 31,
    'Swan': 32,
    'Polar bear': 33,
    'Rabbit': 34,
    'Rhinoceros': 35,
    'Lion': 36,
    'Goat': 37,
    'Centipede': 38,
    'Pig': 39,
    'Sea lion': 40,
    'Zebra': 41,
    'Woodpecker': 42,
    'Elephant': 43,
    'Mouse': 44,
    'Fox': 45,
    'Ostrich': 46,
    'Goldfish': 47,
    'Cheetah': 48,
    'Worm': 49,
    'Leopard': 50,
    'Canary': 51,
    'Brown bear': 52,
    'Crocodile': 53,
    'Raccoon': 54,
    'Jaguar': 55,
    'Sheep': 56,
    'Kangaroo': 57,
    'Panda': 58,
    'Bear': 59,
    'Turkey': 60,
    'Hedgehog': 61,
    'Lynx': 62,
    'Scorpion': 63,
    'Hippopotamus': 64,
    'Otter': 65,
    'Tick': 66,
    'Cattle': 67,
    'Camel': 68,
    'Hamster': 69,
    'Raven': 70,
    'Magpie': 71,
    'Mule': 72,
    'Koala': 73,
    'Bull': 74,
    'Red panda': 75,
    'Shrimp': 76,
    'Turtle': 77,
    'Squid': 78,
    'Seahorse': 79
}

# Function to preprocess image
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get image from POST request
    image_file = request.files['image']
    if not image_file:
        return jsonify({'error': 'No image file found'}), 400
    image = Image.open(io.BytesIO(image_file.read()))

    # Preprocess image
    processed_image = preprocess_image(image, target_size=(64, 64))

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_class_index)]

    # Return prediction result
    return jsonify({'predicted_class': predicted_class_label}), 200

# Run the Flask app
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
