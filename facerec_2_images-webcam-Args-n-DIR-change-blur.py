import cv2
import face_recognition
import sys
import os
import numpy as np

# Function to extract name from file path
def get_name_from_path(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

# Function to list image files in a given directory
def get_image_files(directory):
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = [os.path.join(directory, f) for f in os.listdir(directory)
                   if os.path.splitext(f)[1].lower() in supported_extensions]
    return image_files

# Check if the directory path is passed
if len(sys.argv) != 2:
    print("Usage: python script.py <directory_path>")
    sys.exit(1)

# Directory path from command-line argument
directory_path = sys.argv[1]

# Get list of image files in the specified directory
image_files = get_image_files(directory_path)
if not image_files:
    print("No image files found in the directory.")
    sys.exit(1)

# Load images and compute face encodings
known_face_encodings = []
known_face_names = []

for image_path in image_files:
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        known_face_encodings.append(face_encodings[0])
        known_face_names.append(get_name_from_path(image_path))

# Initialize the video capture object
cap = cv2.VideoCapture(0)
background_color = (0, 0, 0)  # Initial background color: Black
alpha = 0.5  # Initial transparency level
contour_mode = 'normal'  # Options: 'normal', 'blurred', 'off'

# Loop through the frames of the video capture object
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame from BGR to RGB color space
    rgb_frame = frame[:, :, ::-1]

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Create a mask for the faces
    mask = np.zeros_like(frame)
    blurred_frame = cv2.GaussianBlur(frame, (21, 21), 30)  # Blurred frame for blurred mode

    # Draw rectangles around detected faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        if contour_mode == 'normal':
            cv2.rectangle(mask, (left, top), (right, bottom), (255, 255, 255), -1)
        elif contour_mode == 'blurred':
            face_region = blurred_frame[top:bottom, left:right]
            mask[top:bottom, left:right] = face_region

    if contour_mode != 'off':
        frame = cv2.bitwise_and(frame, mask)

    # Change background color
    background = np.full(frame.shape, background_color, dtype=np.uint8)
    frame = cv2.bitwise_or(frame, background)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Handle key presses
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        background_color = (0, 0, 255)  # Red
    elif key == ord('g'):
        background_color = (0, 255, 0)  # Green
    elif key == ord('w'):
        background_color = (255, 255, 255)  # White
    elif key == ord('b'):
        background_color = (0, 0, 0)  # Black
    elif key == ord('n'):
        contour_mode = 'normal'  # Normal contour mode
    elif key == ord('l'):
        contour_mode = 'blurred'  # Blurred contour mode
    elif key == ord('o'):
        contour_mode = 'off'  # Turn off contours

# Release the resources used by OpenCV
cap.release()
cv2.destroyAllWindows()
