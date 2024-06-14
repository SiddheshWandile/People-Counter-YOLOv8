import cv2
import pandas as pd
from ultralytics import YOLO  # Assuming YOLO is compatible with YOLOv5 models
from tracker import Tracker
import cvzone
import asyncio
from datetime import datetime
import sqlite3

model = YOLO('yolov5su.pt')  

# Load class list
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize variables
count = 0
person_down = {}
tracker = Tracker()
counter_down = []

person_up = {}
counter_up = []
cy1, cy2, offset = 194, 350, 6

# Load video
cap = cv2.VideoCapture('vidp.mp4')

# Shared data variable
shared_data = None

# Function to process video frames


async def process_video():
    global count, person_down, counter_down, person_up, counter_up, cap, shared_data

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 3 != 0:
            await asyncio.sleep(0.01)
            continue

        frame = cv2.resize(frame, (1020, 500))

        results = model.predict(frame)  # Perform prediction using YOLOv5s
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        detections = []

        for index, row in px.iterrows():
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            d = int(row[5])
            c = class_list[d]
            if 'person' in c:
                detections.append([x1, y1, x2, y2])

        bbox_id = tracker.update(detections)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx, cy = int((x3 + x4) / 2), int((y3 + y4) / 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

            # Downside Counter
            if (cy + offset) > cy1 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                person_down[id] = (cx, cy)
            if id in person_down and (cy + offset) > cy2 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                if id not in counter_down:
                    counter_down.add(id)
                    del person_down[id]

            # Upside Counter
            if (cy + offset) > cy2 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                person_up[id] = (cx, cy)
            if id in person_up and (cy + offset) > cy1 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                if id not in counter_up:
                    counter_up.add(id)
                    del person_up[id]

        # Draw lines and counters
        cv2.line(frame, (3, cy1), (1018, cy1), (0, 255, 0), 2)
        cv2.line(frame, (5, cy2), (1019, cy2), (0, 255, 255), 2)

        down = len(counter_down)
        up = len(counter_up)
        log_data_to_db(camera_id, down, up, up + down)

        cvzone.putTextRect(frame, f'Enter: {down}', (50, 60), 2, 2)
        cvzone.putTextRect(frame, f'Exit: {up}', (50, 100), 2, 2)
        cvzone.putTextRect(frame, f'Total: {up + down}', (800, 60), 2, 2)

        cv2.imshow(f"Camera {camera_id}", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        await asyncio.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

# Main function to start video processing
async def main():
    video_paths = ['stock-footage.webm']

    video_tasks = [asyncio.create_task(process_video(camera_id, path)) for camera_id, path in enumerate(video_paths)]
    await asyncio.gather(*video_tasks)

# Run the main function
asyncio.run(main())
