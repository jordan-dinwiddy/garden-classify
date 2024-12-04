import tensorflow as tf
import numpy as np
import os

# Log function for consistency
def log(message):
    print(f"[INFERENCE] {message}")

# Parameters
img_size = (256, 256)
model_save_path = "garden_classifier_model.h5"

# Ensure the model exists
if not os.path.exists(model_save_path):
    log(f"Model file {model_save_path} not found. Train the model first.")
    exit(1)

# Load the model
log(f"Loading the model from {model_save_path}...")
model = tf.keras.models.load_model(model_save_path)
log("Model loaded successfully.")

# Create a new dummy image for inference
log("Generating a new test image...")
new_image = np.random.rand(1, img_size[0], img_size[1], 3)  # Replace with an actual image for real inference

# Run inference
log("Running inference on the test image...")
predictions = model.predict(new_image)
log(f"Predicted probabilities for chair, person, dog/pet: {predictions[0]}")