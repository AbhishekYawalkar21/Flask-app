from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.lite.python.interpreter import Interpreter
import numpy as np

app = Flask(__name__)

# Load the trained model (replace 'model.tflite' with the path to your saved model)
interpreter = Interpreter(model_path='model_best.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Manually define the class indices based on your training dataset
class_indices = {
    'Tomato___Bacterial_spot': 0,
    'Tomato___Early_blight': 1,
    'Tomato___Late_blight': 2,
    'Tomato___Leaf_Mold': 3,
    'Tomato___Septoria_leaf_spot': 4,
    'Tomato___Spider_mites Two-spotted_spider_mite': 5,
    'Tomato___Target_Spot': 6,
    'Tomato_Yellow_Leaf_Curl_Virus': 7,
    'Tomato_mosaic_virus': 8,
    'Tomato___healthy': 9
}

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Tomato Crop Disease Identification Flask App!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the POST request
    image_data = request.files['image']
    
    # Save the image to a temporary file
    image_path = 'temp.jpg'
    image_data.save(image_path)

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
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

    # Delete the temporary image file
    os.remove(image_path)

    return jsonify({'prediction': predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)
