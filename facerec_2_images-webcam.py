import os
import cv2
from face_recognition import face_locations, compare_faces
from face_recognition import face_landmarks

# Load the pre-trained model and images
model = face_locations('model')
images = [face_locations('image1'), face_locations('image2')]

# Initialize the video capture object to display the results
cap = cv2.VideoCapture(0)

# Loop through the frames of the video capture object
while True:
    ret, frame = cap.read()
    
    # If no frame was captured, break out of the loop
    if not ret:
        break
    
    # Convert the frame to grayscale and apply equalization
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Detect faces in the frame using the face detection algorithm
    faces = face_landmarks(gray, model=model)[0]['faces']
    
    # Compare the key points of each face to those of the pre-trained model
    matches = []
    for face in faces:
        if compare_faces(face, model):
            matches.append(face)
        
    # Calculate the percentage of matching key points
    percent_matching = len(matches) / len(faces) * 100
    
    # Display the percentage of matching key points and the images with tags and key points
    cv2.putText(frame, f"Percentage of Matching Key Points: {percent_matching}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    for i, match in enumerate(matches):
        cv2.rectangle(frame, (match['x'], match['y']), (match['x'] + match['w'], match['y'] + match['h']), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {i}", (match['x'], match['y']))
    # Save the original and matched images with tags and key points
    os.system("ffmpeg -i image1 -c copy output1.png")
    os.system("ffmpeg -i image2 -c copy output2.png")

# Release the resources used by OpenCV
cv2.destroyAllWindows()
cap.release()
