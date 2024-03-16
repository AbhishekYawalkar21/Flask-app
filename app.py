from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.lite.python.interpreter import Interpreter
from io import BytesIO
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model (replace 'model.tflite' with the path to your saved model)
interpreter = Interpreter(model_path='model_best.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Manually define the class indices based on your training dataset
class_indices = {
    'Bacterial Spot': 0,
    'Early Blight': 1,
    'Late Blight': 2,
    'Leaf Mold': 3,
    'Septoria leaf spot': 4,
    'Two-spotted spider mite': 5,
    'Target_Spot': 6,
    'TomatoYellow Leaf Curl Virus': 7,
    'Tomato Mosaic Virus': 8,
    'Healthy': 9
}

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Tomato Crop Disease Identification Flask App!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Read the image data and create a BytesIO object
    image_data = BytesIO(image_file.read())

    # Load and preprocess the image
    img = image.load_img(image_data, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    img_preprocessed = img_array_expanded_dims / 255.0

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], img_preprocessed)

    # Run the inference
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction, axis=1)

    # Invert the class indices dictionary to map from index to class name
    inverse_map = {v: k for k, v in class_indices.items()}
    predicted_class_name = inverse_map[predicted_class_index[0]]

    return jsonify({'prediction': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)