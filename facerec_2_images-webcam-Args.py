import cv2
import face_recognition
import sys
import os

# Function to extract name from file path
def get_name_from_path(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

# Check if the correct number of arguments is passed
if len(sys.argv) != 3:
    print("Usage: python script.py <image1> <image2>")
    sys.exit(1)

# Load images from command-line arguments
image1_path = sys.argv[1]
image2_path = sys.argv[2]
image1 = face_recognition.load_image_file(image1_path)
image2 = face_recognition.load_image_file(image2_path)

# Compute face encodings
image1_encoding = face_recognition.face_encodings(image1)[0]
image2_encoding = face_recognition.face_encodings(image2)[0]

# Known face encodings and their respective labels derived from file names
known_face_encodings = [image1_encoding, image2_encoding]
known_face_names = [get_name_from_path(image1_path), get_name_from_path(image2_path)]

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
