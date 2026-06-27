import cv2
import os
import numpy as np
from datetime import datetime

if not os.path.exists('faces'):
    os.makedirs('faces')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(1)
print("Camera open! Press 'S' to save your face, 'Q' to quit")

count = 0
name = input("Enter your name: ")

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detected!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow('Attendance System', frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        count += 1
        filename = f'faces/{name}_{count}.jpg'
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
    if key == ord('q') or count >= 5:
        break

cam.release()
cv2.destroyAllWindows()
print(f"Done! {count} images saved for {name}")