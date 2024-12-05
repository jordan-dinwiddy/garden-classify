import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img, save_img

# Input and output directories
INPUT_DIR = "dataset/images"
OUTPUT_DIR = "dataset/augmented_images"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of augmentations per image
AUGMENTATIONS_PER_IMAGE = 10

# Define data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),  # Randomly flip images
    tf.keras.layers.RandomRotation(0.1),  # Rotate by ±20%
    tf.keras.layers.RandomZoom(0.1),      # Zoom in/out by 10%
    tf.keras.layers.RandomBrightness(0.2),  # Adjust brightness by ±20%
    tf.keras.layers.RandomContrast(0.2)  # Adjust contrast by ±20%
])

# Augment and save images
for img_name in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, img_name)
    img = load_img(img_path)  # Load image
    img_array = img_to_array(img)  # Convert to array
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

    for i in range(AUGMENTATIONS_PER_IMAGE):
        augmented_img = data_augmentation(img_array)  # Apply augmentation
        augmented_img = tf.squeeze(augmented_img, axis=0)  # Remove batch dimension

        # Save the augmented image
        output_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i+1}.jpg"
        output_img_path = os.path.join(OUTPUT_DIR, output_img_name)
        save_img(output_img_path, augmented_img.numpy())
        print(f"Saved: {output_img_path}")
        