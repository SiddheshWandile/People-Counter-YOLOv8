# People Counter Using YOLOv8

This project implements a people counter using the YOLOv8 model for object detection and a custom tracking algorithm to track the movement of people across a designated area in a video. The application is capable of counting the number of people entering and exiting a specific region within the video frame.

## Features

- Real-time people detection using YOLOv8.
- Object tracking with unique ID assignment.
- Counting people entering and exiting a designated area.
- Display of total counts on the video feed.

## Requirements

- Python 3.7+
- OpenCV
- pandas
- cvzone
- ultralytics (YOLOv8)
- Custom tracker module (`tracker.py`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SiddheshWandile/People-Counter-YOLOv8.git
   cd People-Counter-YOLOv8
2. Create a virtual environment and activate it:
   ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On Unix or MacOS
    source .venv/bin/activate

# Usage
1. Ensure you have the following files in the project directory:
- vidp.mp4: The input video file.
- coco.txt: The file containing class names for the YOLO model.
- yolov8s.pt: The pre-trained YOLOv8 model weights.
- tracker.py: The custom tracker module.

2. Run the main script:
    ```bash
    python main.py
 
# Main Script (main.py)
The main script performs the following steps:

1. Loads the YOLOv8 model.
2. Sets up mouse callback for the video window (for future extensions).
3. Loads the input video and class names.
4. Initializes the custom tracker.
5. Processes each frame of the video:
- Resizes the frame.
- Runs object detection.
- Tracks detected objects.
- Counts people entering and exiting a designated area.
- Displays the counts on the video feed.
6. Displays the processed video with annotations.

# Tracker Module (tracker.py)
The Tracker class keeps track of the detected objects by their center points and assigns unique IDs to each object. It updates the positions of these objects in each frame and removes objects that are no longer detected.

# Acknowledgements
- YOLOv8
- OpenCV
- cvzone
