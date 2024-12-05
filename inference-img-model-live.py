import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import time


# Define the labels (same order as during training)
LABELS = ["starbucks", "person", "dog"]

# Load the trained model
MODEL_PATH = "models/img_model-transfer.a.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

# Preprocess a single frame
def preprocess_frame(frame, target_size=(128, 128)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, target_size)  # Resize to model input size
    frame_normalized = frame_resized / 255.0  # Normalize to [0, 1]
    return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension

# Perform inference on a single frame
def predict_frame(frame):
    processed_frame = preprocess_frame(frame)
    predictions = model.predict(processed_frame)[0]  # Get the first (and only) prediction
    print(predictions)
    results = {LABELS[i]: predictions[i] for i in range(len(LABELS))}  # Map predictions to labels
    return results

# Main loop for webcam inference
def run_webcam_inference():
    cap = cv2.VideoCapture(0)  # Open the webcam
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from webcam.")
            break

        # Run inference on the current frame
        predictions = predict_frame(frame)

        # Print probabilities to stdout
        for label, probability in predictions.items():
            if probability > 0.5:  # Check if probability is above threshold
                print(f"  {label}")

        # Print probabilities to stdout
        print("\nPredictions:")
        for label, probability in predictions.items():
            print(f"  {label}: {probability:.2f}")

        # Display the frame in a window (optional)
        cv2.imshow("Webcam Inference", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Sleep to limit the loop frequency
        time.sleep(0.5)

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_inference()