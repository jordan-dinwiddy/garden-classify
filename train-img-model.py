import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

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

def preprocess_image(image_path, target_size=(128, 128)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)  # Resize to model input size
    image = image / 255.0  # Normalize to [0, 1]
    return image

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

# Combine original and augmented data
def combine_data(df, original_dir, augmented_dir):
    # Add paths for original images
    original_paths = [os.path.join(original_dir, img) for img in df["Image Name"]]
    original_targets = df["Target"].tolist()

    # Add paths for augmented images (assume they share the same labels as originals)
    augmented_paths = []
    augmented_targets = []
    for img, target in zip(df["Image Name"], original_targets):
        base_name, ext = os.path.splitext(img)
        aug_files = [f"{base_name}_aug_{i+1}{ext}" for i in range(20)]  # Assuming 20 augmentations
        for aug_file in aug_files:
            augmented_paths.append(os.path.join(augmented_dir, aug_file))
            augmented_targets.append(target)

    # Combine original and augmented data
    all_paths = original_paths + augmented_paths
    all_targets = original_targets + augmented_targets

    return pd.DataFrame({"Path": all_paths, "Target": all_targets})


# Create TensorFlow dataset
def create_dataset(df):
    paths = df["Path"].tolist()
    targets = df["Target"].tolist()

    dataset = tf.data.Dataset.from_tensor_slices((paths, targets))
    dataset = dataset.map(
        lambda x, y: (preprocess_image(x), tf.cast(y, dtype=tf.float32))  # Ensure labels are float32
    )
    return dataset


# Parameters
model_save_path = "models/img_model.h5"
tflite_save_path = "models/img_model.tflite"


# Load the labels
df_labels = load_labels(LABELS_FILE)
combined_df = combine_data(df_labels, ORIGINAL_DIR, AUGMENTED_DIR)

# Split into training and validation sets
train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42, stratify=[str(t) for t in combined_df["Target"]])

# Create training and validation datasets
train_dataset = create_dataset(train_df).shuffle(1000).batch(32)
val_dataset = create_dataset(val_df).batch(32)

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
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,  # You can increase epochs for larger datasets
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# Save the model
log(f"Saving the model to {model_save_path}...")
model.save(model_save_path)


log("Model training and saving complete.")