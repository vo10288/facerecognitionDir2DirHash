import face_recognition
import cv2
import sys
import os
import numpy as np
import csv
from datetime import datetime
import hashlib

def get_image_files(directory):
    """ Return a list of image files in the specified directory. """
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in supported_extensions]

def file_hash(filepath):
    """ Generate MD5 hash of a file. """
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        chunk = f.read(8192)
        while chunk:
            md5.update(chunk)
            chunk = f.read(8192)
    return md5.hexdigest()

def save_to_csv(file_name, results):
    """ Save the face comparison results to a CSV file. """
    fieldnames = ['Image_1', 'Image_2', 'Image_1_Hash', 'Image_2_Hash', 'Match_Result', 'Match_Percentage']
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def main(directory_path1, directory_path2):
    """ Process each image in directory_path1 against each image in directory_path2. """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    image_files1 = get_image_files(directory_path1)
    image_files2 = get_image_files(directory_path2)
    results = []

    for image_file1 in image_files1:
        image1 = face_recognition.load_image_file(image_file1)
        image1_encodings = face_recognition.face_encodings(image1)
        if not image1_encodings:
            print(f"No faces found in {image_file1}")
            continue
        image1_hash = file_hash(image_file1)

        for image_file2 in image_files2:
            image2 = face_recognition.load_image_file(image_file2)
            image2_encodings = face_recognition.face_encodings(image2)
            if not image2_encodings:
                print(f"No faces found in {image_file2}")
                continue
            image2_hash = file_hash(image_file2)

            match = face_recognition.compare_faces([image1_encodings[0]], image2_encodings[0], tolerance=0.6)
            distance = face_recognition.face_distance([image1_encodings[0]], image2_encodings[0])
            match_percentage = (1 - distance[0]) * 100

            result_info = {
                'Image_1': image_file1,
                'Image_2': image_file2,
                'Image_1_Hash': image1_hash,
                'Image_2_Hash': image2_hash,
                'Match_Result': match[0],
                'Match_Percentage': f"{match_percentage:.2f}%"
            }
            results.append(result_info)
            print(f"Comparing {os.path.basename(image_file1)} to {os.path.basename(image_file2)}: "
                  f"Match = {match[0]}, {match_percentage:.2f}% match")

            # Tag and show images
            tagged_image1 = convert_to_bgr(image1)
            draw_on_image(tagged_image1, face_recognition.face_locations(image1), face_recognition.face_landmarks(image1))
            cv2.imshow(f"Tagged {os.path.basename(image_file1)}", tagged_image1)
            cv2.waitKey(500)  # Display for 500 milliseconds
            cv2.imwrite(os.path.join(results_dir, f"tagged_{os.path.basename(image_file1)}"), tagged_image1)

            tagged_image2 = convert_to_bgr(image2)
            draw_on_image(tagged_image2, face_recognition.face_locations(image2), face_recognition.face_landmarks(image2))
            cv2.imshow(f"Tagged {os.path.basename(image_file2)}", tagged_image2)
            cv2.waitKey(500)  # Display for 500 milliseconds
            cv2.imwrite(os.path.join(results_dir, f"tagged_{os.path.basename(image_file2)}"), tagged_image2)

            cv2.destroyAllWindows()

    # Save results to CSV
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
        print("Usage: python script.py <directory_path1> <directory_path2>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
