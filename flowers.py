from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Load the model
model = load_model("models/flowers/keras_model.h5", compile=False)

# Load the labels
class_names = open("models/flowers/labels.txt", "r").readlines()

# Create the `flowers` function
def flowers(file):
    try:
        # Define the input size
        size = (224, 224)

        # Open and preprocess the image
        image = Image.open(file).convert("RGB")
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Prepare the input data for the model
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Perform prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = float(prediction[0][index])

        # Return the result
        return class_name, confidence_score
    except Exception as e:
        print(f"Error in flowers function: {e}")
        return None  # Return None if an error occurs
