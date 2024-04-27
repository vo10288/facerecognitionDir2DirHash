import face_recognition
import cv2
import sys
import numpy as np

# Check if the correct number of arguments is passed
if len(sys.argv) != 3:
    print("Usage: python script.py <image1_path> <image2_path>")
    sys.exit(1)

# Load the images from command line arguments
image1_path = sys.argv[1]
image2_path = sys.argv[2]
image1 = face_recognition.load_image_file(image1_path)
image2 = face_recognition.load_image_file(image2_path)

# Find the face locations and face landmarks in each image
face_locations1 = face_recognition.face_locations(image1)
face_landmarks1 = face_recognition.face_landmarks(image1)
face_locations2 = face_recognition.face_locations(image2)
face_landmarks2 = face_recognition.face_landmarks(image2)

# Function to convert an image from RGB (face_recognition format) to BGR (OpenCV format)
def convert_to_bgr(image):
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Prepare the images for showing
image1_display = convert_to_bgr(image1)
image2_display = convert_to_bgr(image2)

# Function to add the face landmarks and rectangles to the image
def draw_on_image(image, face_locations, face_landmarks_list):
    if image is not None:
        for face_location, face_landmarks in zip(face_locations, face_landmarks_list):
            # Draw a rectangle around the face
            top, right, bottom, left = face_location
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw all the face landmarks
            for facial_feature in face_landmarks.keys():
                points = face_landmarks[facial_feature]
                for point in points:
                    cv2.circle(image, point, 2, (0, 0, 255), -1)

# Draw on the images
draw_on_image(image1_display, face_locations1, face_landmarks1)
draw_on_image(image2_display, face_locations2, face_landmarks2)

# Show the images
cv2.imshow('Image 1', image1_display)
cv2.imshow('Image 2', image2_display)

# Encode the faces from both images
if face_locations1 and face_locations2:
    encoding1 = face_recognition.face_encodings(image1, [face_locations1[0]])[0]
    encoding2 = face_recognition.face_encodings(image2, [face_locations2[0]])[0]

    # Compare the faces
    results = face_recognition.compare_faces([encoding1], encoding2)
    print("Are the faces the same person? ", results[0])
else:
    print("Face not detected in one of the images.")

# Wait until a key is pressed to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
