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
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']  # Add or remove file types as needed
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
    if face_encodings:  # Check if face is detected
        known_face_encodings.append(face_encodings[0])
        known_face_names.append(get_name_from_path(image_path))

# Initialize the video capture object
cap = cv2.VideoCapture(0)
background_color = (0, 0, 0)  # Initial background color: Black
alpha = 0.5  # Initial transparency level

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

    # Create a black image of the same size as the frame to mask everything
    mask = np.zeros_like(frame)

    # Compare each detected face to known faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the first match found in the known face encodings
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face and label it
        cv2.rectangle(mask, (left, top), (right, bottom), (255, 255, 255), -1)  # Fill the face area with white

    # Blend the original frame with the mask
    frame = cv2.addWeighted(frame, alpha, mask, 1 - alpha, 0)

    # Apply the mask to the frame
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
    elif key == ord('+') and alpha < 1:
        alpha += 0.1  # Increase contour transparency
    elif key == ord('-') and alpha > 0:
        alpha -= 0.1  # Decrease contour transparency

# Release the resources used by OpenCV
cap.release()
cv2.destroyAllWindows()
