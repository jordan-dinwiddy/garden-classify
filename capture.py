import cv2
import os
import csv
from PIL import Image

# Directory for storing images and CSV file for labels
IMAGE_DIR = "dataset/images"
LABEL_FILE = "dataset/labels.csv"

# Available labels
LABELS = ["starbucks", "person", "dog", "none"]

# Initial window size
WINDOW_WIDTH = 320
WINDOW_HEIGHT = 240

# Ensure image directory exists
os.makedirs(IMAGE_DIR, exist_ok=True)

# Initialize the CSV file if it doesn't exist
if not os.path.exists(LABEL_FILE):
    with open(LABEL_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Labels"])  # Header row

# Save the captured frame with multiple labels
def save_image_with_metadata(frame, labels):
    # Convert BGR (OpenCV format) to RGB (Pillow format)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    # Generate a unique filename
    filename = f"image_{len(os.listdir(IMAGE_DIR)) + 1}.jpg"
    filepath = os.path.join(IMAGE_DIR, filename)

    # Save the image
    pil_image.save(filepath)
    print(f"Saved: {filepath}")

    # Update metadata file with labels
    with open(LABEL_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([filename, ",".join(labels)])
    print(f"Updated labels: {filename} -> {labels}")

# Main program for capturing images
def capture_images():
    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    current_labels = []
    default_window_title = "Camera - Press 'l' to select labels, 'c' to capture, 'q' to quit"
    cv2.namedWindow(default_window_title, cv2.WINDOW_NORMAL)  # Enable manual resizing
    cv2.resizeWindow(default_window_title, WINDOW_WIDTH, WINDOW_HEIGHT)  # Force specific size

    def update_window_title(labels=None):
        """Update the title bar of the OpenCV window."""
        label_text = ", ".join(labels) if labels else "None"
        title = f"{default_window_title} (Current Labels: {label_text})"
        cv2.setWindowTitle(default_window_title, title)

    print("Press 'l' to select labels, 'c' to capture an image, and 'q' to quit.")

    while True:
        # Capture a frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Display the frame without resizing
        cv2.imshow(default_window_title, frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Quit
            print("Exiting.")
            break
        elif key == ord('l'):  # Select labels
            print(f"Available labels: {', '.join(LABELS)}")
            selected_labels = input("Enter labels separated by commas: ").strip().lower().split(",")
            selected_labels = [label.strip() for label in selected_labels if label.strip() in LABELS]
            if selected_labels:
                current_labels = selected_labels
                print(f"Labels set to: {current_labels}")
                update_window_title(current_labels)
            else:
                print(f"Invalid labels. Please choose from {LABELS}.")
        elif key == ord('c'):  # Capture image
            if not current_labels:
                print("No labels selected. Press 'l' to choose labels.")
            else:
                save_image_with_metadata(frame, current_labels)

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images()