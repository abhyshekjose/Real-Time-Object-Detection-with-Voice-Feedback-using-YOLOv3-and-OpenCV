Real-Time Object Detection with Voice Feedback using YOLOv3 and OpenCV

This project implements real-time object detection using the YOLOv3 deep learning model and provides audio feedback using a text-to-speech engine.

Features

Real-time video feed processing from webcam

Object detection using YOLOv3 (You Only Look Once)

Labels and bounding boxes drawn on detected objects

Spoken output of detected objects using pyttsx3

Avoids repeated announcements for the same detected object

Project Structure

Blind/
├── app.py
├── yolo-coco/
│   ├── yolov3.cfg
│   ├── yolov3.weights
│   └── coco.names

Requirements

Ensure Python version is 3.6 to 3.12 (not 3.13).

Install the required packages:

pip install numpy opencv-python pyttsx3 imutils

YOLOv3 Files Needed

Download the following files and place them in the yolo-coco/ folder:

yolov3.cfg

yolov3.weights (approx. 248MB)

coco.names

How to Run

Ensure all files are set up correctly.

Open a terminal in the project folder.

Run the application:

python app.py

The webcam window will display the live feed with detected object boxes and labels.

When an object is detected, its label will be spoken aloud.

Press q to exit.

Text-to-Speech (TTS)

This project uses pyttsx3, which works offline and does not require internet access. It is platform-independent and works on Windows, Linux, and macOS.

Exit Instructions

To stop the application, press the q key in the video window.

Acknowledgments

YOLOv3 by Joseph Redmon (https://pjreddie.com/darknet/yolo/)

OpenCV library for computer vision tasks

pyttsx3 for offline speech synthesis
