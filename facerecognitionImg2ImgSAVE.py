import face_recognition
import cv2
import sys
import numpy as np
import os
import csv
from datetime import datetime

# Create a directory based on the current timestamp
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(f"results_{timestamp}", exist_ok=True)

def save_to_csv(file_name, face_locations, face_landmarks_list, match_results=None):
    fieldnames = ['Face', 'Location', 'Landmarks', 'Match_Result']
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, (location, landmarks) in enumerate(zip(face_locations, face_landmarks_list)):
            writer.writerow({
                'Face': i + 1,
                'Location': location,
                'Landmarks': {k: [tuple(point) for point in v] for k, v in landmarks.items()},
                'Match_Result': match_results if match_results and i == 0 else ''
            })

if len(sys.argv) != 3:
    print("Usage: python script.py <image1_path> <image2_path>")
    sys.exit(1)

image1_path, image2_path = sys.argv[1:3]
image1 = face_recognition.load_image_file(image1_path)
image2 = face_recognition.load_image_file(image2_path)

face_locations1 = face_recognition.face_locations(image1)
face_landmarks1 = face_recognition.face_landmarks(image1)
face_locations2 = face_recognition.face_locations(image2)
face_landmarks2 = face_recognition.face_landmarks(image2)

def convert_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image is not None else None

image1_display = convert_to_bgr(image1)
image2_display = convert_to_bgr(image2)

def draw_on_image(image, face_locations, face_landmarks_list):
    if image is not None:
        for face_location, face_landmarks in zip(face_locations, face_landmarks_list):
            top, right, bottom, left = face_location
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            for facial_feature in face_landmarks.keys():
                points = face_landmarks[facial_feature]
                for point in points:
                    cv2.circle(image, tuple(point), 2, (0, 0, 255), -1)

draw_on_image(image1_display, face_locations1, face_landmarks1)
draw_on_image(image2_display, face_locations2, face_landmarks2)

cv2.imwrite(f"results_{timestamp}/image1_tagged.jpg", image1_display)
cv2.imwrite(f"results_{timestamp}/image2_tagged.jpg", image2_display)

match_results = "No faces detected"
if face_locations1 and face_locations2:
    encoding1 = face_recognition.face_encodings(image1, [face_locations1[0]])[0]
    encoding2 = face_recognition.face_encodings(image2, [face_locations2[0]])[0]
    results = face_recognition.compare_faces([encoding1], encoding2, tolerance=0.6)
    face_distances = face_recognition.face_distance([encoding1], encoding2)
    match_percentage = (1 - face_distances[0]) * 100
    match_results = f"{match_percentage:.2f}% match"
    print("Face comparison result:", match_results)

csv_file_path1 = f"results_{timestamp}/image1_results.csv"
csv_file_path2 = f"results_{timestamp}/image2_results.csv"
save_to_csv(csv_file_path1, face_locations1, face_landmarks1, match_results)
save_to_csv(csv_file_path2, face_locations2, face_landmarks2)

cv2.imshow('Image 1 Tagged', image1_display)
cv2.imshow('Image 2 Tagged', image2_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
