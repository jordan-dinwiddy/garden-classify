import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import pandas as pd

# Paths
LABELS_FILE = "dataset/labels.csv"
ORIGINAL_DIR = "dataset/images"
AUGMENTED_DIR = "dataset/augmented_images"

LABELS = ["starbucks", "person", "dog"]

# Load and process labels.csv, return essentially a map of images name -> label + target vector
def load_labels(labels_file):
    # Read the CSV into a DataFrame
    df = pd.read_csv(labels_file)

    # Convert labels to multi-hot vectors
    def labels_to_vector(label_str):
        vector = [0] * len(LABELS)  # Initialize zero vector
        if label_str != "none":  # Skip "none" labels
            for label in label_str.split(","):
                label = label.strip()
                if label in LABELS:
                    vector[LABELS.index(label)] = 1
        return vector

    df["Target"] = df["Labels"].apply(labels_to_vector)
    return df

# TensorFlow dataset creation
def preprocess_image(image_path, target):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128, 128])  # Resize to model input size
    image = image / 255.0  # Normalize to [0, 1]
    target = tf.cast(target, dtype=tf.float32)  # Cast target to float32
    return image, target

# Log function for consistency
def log(message):
    print(f"[TRAINING] {message}")

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
model_save_path = "models/img_model.h5"
tflite_save_path = "models/img_model.tflite"


# Load the labels
df_labels = load_labels(LABELS_FILE)

# Create paths and targets for the original images
original_paths = [os.path.join(ORIGINAL_DIR, img) for img in df_labels["Image Name"]]
original_targets = df_labels["Target"].tolist()

# Augment images and duplicate their labels
augmented_paths = []
augmented_targets = []
for img_path, target in zip(original_paths, original_targets):
    for i in range(20):  # Number of augmentations per image
        augmented_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_aug_{i+1}.jpg"
        augmented_paths.append(os.path.join(AUGMENTED_DIR, augmented_name))
        augmented_targets.append(target)  # Same target as the original

# Combine original and augmented data
all_paths = original_paths + augmented_paths
all_targets = original_targets + augmented_targets

# Loop over all paths and print path -> target
for path, target in zip(all_paths, all_targets):
    print(f"{path} -> {target}")

dataset = tf.data.Dataset.from_tensor_slices((all_paths, all_targets))
dataset = dataset.map(preprocess_image).batch(32).shuffle(1000)

# Build the model
log("Building the model...")
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(LABELS), activation='sigmoid')  # Sigmoid for multi-label classification
])

# Compile the model
log("Compiling the model...")
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary cross-entropy for multi-label
              metrics=['accuracy'])

# Train the model
log("Starting training...")
model.fit(dataset, epochs=20)

# Save the model
log(f"Saving the model to {model_save_path}...")
model.save(model_save_path)


log("Model training and saving complete.")