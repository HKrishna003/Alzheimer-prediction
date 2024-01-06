import tensorflow as tf
import numpy as np
from PIL import Image
def load_and_preprocess_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Resize the image to match the input shape
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values to the range [0, 1]

    # Ensure the image has 3 channels (RGB)
    if img_array.shape[-1] != 3:
        img_array = np.stack((img_array,) * 3, axis=-1)

    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_alzheimer(image_path):
    # Load the saved model
    model = tf.keras.models.load_model("C:/Users/harik/OneDrive/Desktop/Data Science/Alz test/Alzheimer Classifier")

    # Preprocess the input image
    processed_image = load_and_preprocess_image(image_path)

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

# Specify the file path of the image you want to test
image_path_to_test = 'mildDem9.jpg'

# Call the predict_alzheimer function
prediction_result = predict_alzheimer(image_path_to_test)

# Print the prediction result
print("Predicted probabilities:", prediction_result)

# Call the function to get the predicted class
predicted_class = get_predicted_class(prediction_result)
print("Predicted class:", predicted_class)
