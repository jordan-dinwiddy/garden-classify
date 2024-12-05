import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define the labels (same order as during training)
LABELS = ["starbucks", "person", "dog"]

# Load the trained model
MODEL_PATH = "models/img_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

# Preprocess a single image
def preprocess_image(image_path, target_size=(128, 128)):
    image = load_img(image_path, target_size=target_size)  # Load and resize the image
    image_array = img_to_array(image)  # Convert to array
    image_array = image_array / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Perform inference on a single image
def predict_image(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)[0]  # Get the first (and only) prediction
    results = {LABELS[i]: predictions[i] for i in range(len(LABELS))}  # Map predictions to labels
    return results

# Example usage
if __name__ == "__main__":
    IMAGE_PATH = "dataset/images/image_43.jpg"  # Replace with your image path
    predictions = predict_image(IMAGE_PATH)
    print(f"Predictions for {IMAGE_PATH}:")
    for label, probability in predictions.items():
        print(f"  {label}: {probability:.2f}")