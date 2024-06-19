# Human_Detection


In this project we are working on detecting the human beings in the cctv footage at real time.</br>
This involves using computer vision techniques and machine learning models to identify and track people in video streams.


## Prerequisites
Before running the code, ensure you have the following dependencies installed:

- OpenCV (cv2)
- Ultralytics YOLO (ultralytics)
- cvzone (cvzone)
- Python 3.7+


## Installation
To run this code, ensure you have the necessary libraries installed. You can install them using pip:

```python
pip install opencv-python cvzone ultralytics
```


## Explanation

### Load YOLO Model
The YOLOv8 model is loaded using the ultralytics library:

```python
from ultralytics import YOLO
model = YOLO("../Yolo-Weights/yolov8l.pt")
```

### Class Names
A list of class names is defined for the model to interpret the detected objects:

```python
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

```

### Initialize Image Capture
Image capture is initialized using OpenCV. Here you need to put your Image datset in the ("/content/0.png") section:

```python
import cv2
import cvzone

# For webcam, use cv2.VideoCapture(0) and adjust the index if necessary
cap = cv2.VideoCapture("/content/0.png")  # RTSP stream URL
cap.set(3, 1280)  # Set frame width
cap.set(4, 720)  # Set frame height
```

### Main Loop
The main loop captures frames, processes them through the YOLO model, and annotates detected persons:
```python
import math
import time

prev_frame_time = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Run model on the image
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if classNames[cls] == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100

                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'Person {conf}', (max(0, x1), max(1, y1)), scale=1, thickness=1)

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps}")

    # Display the processed image inline
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Break after displaying the image once
    break

# Release capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
```

### Explanation of Main Loop
- Frame Capture: Reads frames from the video stream.
- Model Inference: Passes each frame to the YOLO model for detection.
- Bounding Box and Annotation: If a "person" is detected, draws a bounding box and confidence score.
- FPS Calculation: Computes and prints the frames per second (FPS) to monitor performance.
- Display: Shows the annotated frame in a window.

##

This code sets up a real-time person detection system using YOLOv8 and OpenCV, ideal for applications requiring object detection and tracking.

##
<br>I hope this explanation was helpful to you.</br>
Thank you for your support.</br>
Keep Learning.</br>
~Pranjali Goyal</br>
