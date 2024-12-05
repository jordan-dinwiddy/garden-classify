import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import models, layers, optimizers


# Paths
LABELS_FILE = "dataset/labels.csv"
ORIGINAL_DIR = "dataset/images"
AUGMENTED_DIR = "dataset/augmented_images"

LABELS = ["starbucks", "person", "dog"]

AUGMENTATIONS_PER_IMAGE = 10

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
        aug_files = [f"{base_name}_aug_{i+1}{ext}" for i in range(AUGMENTATIONS_PER_IMAGE)]  # Assuming n augmentations
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
model_save_path = "models/img_model-transfer.a.h5"

# Load the labels
df_labels = load_labels(LABELS_FILE)
combined_df = combine_data(df_labels, ORIGINAL_DIR, AUGMENTED_DIR)

# Split into training and validation sets
train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42, stratify=[str(t) for t in combined_df["Target"]])

# Create training and validation datasets
train_dataset = create_dataset(train_df).shuffle(1000).batch(32)
val_dataset = create_dataset(val_df).batch(32)


# Load the pretrained MobileNetV2 model
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze the base model initially


# Build the model
log("Building the model...")

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Convert feature maps to a single vector
    layers.Dense(256, activation="relu"),  # Fully connected layer
    layers.Dropout(0.5),  # Regularization
    layers.Dense(len(LABELS), activation="sigmoid")  # Multi-label classification output
])

# Compile the model
log("Compiling the model...")
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the top layers
log("Starting training...")
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# Fine-tune by unfreezing some layers
log("Fine-tine training...")
base_model.trainable = True
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# Save the model
log(f"Saving the model to {model_save_path}...")
model.save(model_save_path)


log("Model training and saving complete.")