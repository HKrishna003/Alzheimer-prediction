from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the saved CNN model
model = tf.keras.models.load_model("models/Alz_classifier.h5")

def preprocess_image(file):
    # Open the image file
    img = Image.open(file)

    # Resize the image to match the input shape
    img = img.resize((256, 256))

    # Convert to NumPy array
    img_array = np.array(img)

    # Normalize pixel values to the range [0, 1]
    img_array = img_array.astype('float32') / 255.0

    # Ensure the image has 3 channels (RGB)
    if img_array.shape[-1] != 3:
        img_array = np.stack((img_array,) * 3, axis=-1)

    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_alzheimer(processed_image):
    # Perform inference
    prediction = model.predict(processed_image)

    return prediction.tolist()

def get_predicted_class(prediction_result):
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(prediction_result)

    # Map the index to the corresponding class name
    class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    predicted_class = class_names[predicted_class_index]

    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No selected file")

        processed_image = preprocess_image(file)
        prediction_result = predict_alzheimer(processed_image)
        predicted_class = get_predicted_class(prediction_result)

        return render_template('index.html', prediction=predicted_class)

    return render_template('index.html', error=None, prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
