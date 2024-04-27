import face_recognition
import cv2
import sys
import os
import numpy as np
import csv
from datetime import datetime

def get_image_files(directory):
    """ Return a list of image files in the specified directory. """
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in supported_extensions]

def save_to_csv(file_name, results):
    """ Save the face comparison results to a CSV file. """
    fieldnames = ['Reference_Image', 'Compared_Image', 'Match_Result', 'Match_Percentage']
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def main(reference_image_path, directory_path):
    """ Process each image in the directory against the reference image. """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    reference_image = face_recognition.load_image_file(reference_image_path)
    reference_image_encodings = face_recognition.face_encodings(reference_image)
    if not reference_image_encodings:
        print("No faces found in the reference image.")
        return
    reference_encoding = reference_image_encodings[0]

    image_files = get_image_files(directory_path)
    results = []
    for image_file in image_files:
        image = face_recognition.load_image_file(image_file)
        image_encodings = face_recognition.face_encodings(image)
        if image_encodings:
            match = face_recognition.compare_faces([reference_encoding], image_encodings[0], tolerance=0.6)
            distance = face_recognition.face_distance([reference_encoding], image_encodings[0])
            match_percentage = (1 - distance[0]) * 100
            result_info = {
                'Reference_Image': os.path.basename(reference_image_path),
                'Compared_Image': os.path.basename(image_file),
                'Match_Result': match[0],
                'Match_Percentage': f"{match_percentage:.2f}%"
            }
            results.append(result_info)

            # Print results to the console
            print(f"Comparing {result_info['Reference_Image']} to {result_info['Compared_Image']}: "
                  f"Match = {result_info['Match_Result']}, {result_info['Match_Percentage']}")

            # Tag and save image
            tagged_image = convert_to_bgr(image)
            draw_on_image(tagged_image, face_recognition.face_locations(image), face_recognition.face_landmarks(image))
            cv2.imwrite(os.path.join(results_dir, f"tagged_{os.path.basename(image_file)}"), tagged_image)
        else:
            print(f"No faces found in {image_file}")

    csv_file_path = os.path.join(results_dir, "comparison_results.csv")
    save_to_csv(csv_file_path, results)

def convert_to_bgr(image):
    """ Convert an image from RGB to BGR. """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def draw_on_image(image, face_locations, face_landmarks_list):
    """ Draw rectangles and landmarks on the image. """
    for face_location, face_landmarks in zip(face_locations, face_landmarks_list):
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        for facial_feature in face_landmarks.keys():
            points = face_landmarks[facial_feature]
            for point in points:
                cv2.circle(image, tuple(point), 2, (0, 0, 255), -1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <reference_image_path> <directory_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
