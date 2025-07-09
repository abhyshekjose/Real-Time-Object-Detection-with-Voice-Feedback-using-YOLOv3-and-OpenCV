import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not detected or in use!")
else:
    print("Webcam is working fine!")
cap.release()
