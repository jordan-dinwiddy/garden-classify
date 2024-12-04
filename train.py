import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# Log function for consistency
def log(message):
    print(f"[TRAINING] {message}")

# Generate a dummy dataset
def create_dummy_dataset(num_samples=1000, img_size=(256, 256), num_classes=3):
    images = np.random.rand(num_samples, img_size[0], img_size[1], 3)  # Random images
    labels = np.random.randint(0, 2, (num_samples, num_classes))  # Random binary labels
    return images, labels

# Convert the model to TFLite format
def save_model_as_tflite(model, tflite_model_path):
    # Convert to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_model_path}")

# Parameters
img_size = (256, 256)
num_classes = 3
model_save_path = "models/garden_classifier_model.h5"
tflite_save_path = "models/garden_classifier_model.tflite"

# Load dataset
log("Generating dataset...")
train_images, train_labels = create_dummy_dataset()
val_images, val_labels = create_dummy_dataset(num_samples=200)

# Build the model
log("Building the model...")
model = models.Sequential([
    layers.Input(shape=(img_size[0], img_size[1], 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='sigmoid')  # Sigmoid for multi-label classification
])

# Compile the model
log("Compiling the model...")
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary cross-entropy for multi-label
              metrics=['accuracy'])

# Train the model
log("Starting training...")
model.fit(
    train_images, train_labels,
    epochs=10,
    batch_size=32,
    validation_data=(val_images, val_labels)
)

# Save the model
log(f"Saving the model to {model_save_path}...")
model.save(model_save_path)

log(f"Saving the model (tflite) to {tflite_save_path}.")
save_model_as_tflite(model, tflite_save_path)


log("Model training and saving complete.")