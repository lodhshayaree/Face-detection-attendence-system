import cv2
import numpy as np
import face_recognition
import os
import csv
from datetime import datetime
import subprocess

# Path to your image file
image_path = 'C:/Users/shaya/Downloads/ps image.jpg'

# Extract the name of the image from the path
image_name = os.path.basename(image_path).split('.')[0]

# Load the image
img = cv2.imread(image_path)
imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

# Encode the known image
encodeKnown = face_recognition.face_encodings(imgS)[0]

print('Encoding Complete')

# Directory to save captured face images
captured_faces_dir = 'captured_faces'
os.makedirs(captured_faces_dir, exist_ok=True)

# Path to the CSV file (example using the script's directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, 'attendance.csv')

# Create CSV file and write header if it doesn't exist
if not os.path.isfile(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Time'])

cap = cv2.VideoCapture(0)
known_faces = set()
last_detected_person = None  # Variable to keep track of the last detected person

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Locate faces in the current frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # Compare the faces
        matches = face_recognition.compare_faces([encodeKnown], encodeFace)
        faceDis = face_recognition.face_distance([encodeKnown], encodeFace)
        print(faceDis)

        # Determine color and label based on whether the face matches
        if matches[0]:
            label = image_name
            color = (0, 255, 0)  # Green color for known faces

            # Add the name and timestamp to the CSV file if not already added
            if label not in known_faces and label != last_detected_person:
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([label, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                    known_faces.add(label)
                    last_detected_person = label  # Update last detected person

                    # Open the CSV file using the default application
                    try:
                        subprocess.Popen(['start', csv_file], shell=True)
                    except FileNotFoundError:
                        print(f"Error: Unable to open {csv_file}")

        else:
            label = 'Unknown'
            color = (0, 0, 255)  # Red color for unknown faces

        # Draw rectangles and text on the frame
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, label, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Print the label in the program output
        print(f"Face detected: {label}")

    cv2.imshow('webcam', img)
    if cv2.waitKey(10) == 13:  # Press 'Enter' to exit
        break

cap.release()
cv2.destroyAllWindows()
