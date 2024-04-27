import cv2
import face_recognition
import sys
import os

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

    # Compare each detected face to known faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the first match found in the known face encodings
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources used by OpenCV
cap.release()
cv2.destroyAllWindows()
