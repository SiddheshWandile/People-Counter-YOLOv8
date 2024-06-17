import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone
import asyncio
from pymongo import MongoClient
from datetime import datetime, timedelta

# Load YOLO model
model = YOLO('yolov8s.pt')

# # Function to capture mouse events (currently just printing coordinates)
# def RGB(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE:
#         point = [x, y]
#         print(point)

# # Set up mouse callback
# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)

# Load class list
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Initialize variables
# Add your camera sources here
camera_sources = ['vidp.mp4']
caps = [cv2.VideoCapture(src) for src in camera_sources]

trackers = [Tracker() for _ in camera_sources]
counts = [0] * len(camera_sources)
person_down = [{} for _ in camera_sources]
counter_down = [[] for _ in camera_sources]
person_up = [{} for _ in camera_sources]
counter_up = [[] for _ in camera_sources]
count_enter = 0
count_exit = 0

cy1, cy2, offset = 194, 240, 6

# Shared data variable
shared_data = [None] * len(camera_sources)

# Connect to MongoDB
# Update the URI as needed
client = MongoClient(
    'mongodb+srv://SiddheshWan:Pass123@peoplecount1.ewmaagh.mongodb.net/peoplecount1?retryWrites=true&w=majority')
db = client['people_count_db']
collection = db['people_count']

# Function to process video frames for a specific camera


async def process_video(cam_index):
    global counts, person_down, counter_down, person_up, counter_up, shared_data, count_enter, count_exit

    cap = caps[cam_index]
    tracker = trackers[cam_index]
    frame_skip = 3  # Process every 3rd frame to reduce load

    last_update_time = datetime.now()  # Initialize last_update_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        counts[cam_index] += 1
        if counts[cam_index] % frame_skip != 0:
            await asyncio.sleep(0.01)
            continue

        frame = cv2.resize(frame, (1020, 500))

        results = model.predict(frame, stream=True)
        detections = []

        for result in results:
            for box in result.boxes:
                # Extract the bounding box coordinates safely
                try:
                    # Get the first bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    d = int(box.cls[0])  # Get the class index
                    c = class_list[d]
                    if 'person' in c:
                        detections.append([x1, y1, x2, y2])
                except ValueError as e:
                    print(f"Error unpacking bounding box coordinates: {e}")
                    continue

        bbox_id = tracker.update(detections)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx, cy = int((x3 + x4) / 2), int((y3 + y4) / 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

            # Downside Counter
            if (cy + offset) > cy1 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                person_down[cam_index][id] = (cx, cy)
            if id in person_down[cam_index] and (cy + offset) > cy2 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                if id not in counter_down[cam_index]:
                    counter_down[cam_index].append(id)
                    count_enter += 1

            # Upside Counter
            if (cy + offset) > cy2 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                person_up[cam_index][id] = (cx, cy)
            if id in person_up[cam_index] and (cy + offset) > cy1 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                if id not in counter_up[cam_index]:
                    counter_up[cam_index].append(id)
                    count_exit += 1

        # Draw lines and counters
        cv2.line(frame, (3, cy1), (1018, cy1), (0, 255, 0), 2)
        cv2.line(frame, (5, cy2), (1019, cy2), (0, 255, 255), 2)

        down = len(counter_down[cam_index])
        up = len(counter_up[cam_index])

        cvzone.putTextRect(frame, f'Enter: {down}', (50, 60), 2, 2)
        cvzone.putTextRect(frame, f'Exit: {up}', (50, 100), 2, 2)
        cvzone.putTextRect(frame, f'Total: {up + down}', (800, 60), 2, 2)

        # Check if it's time to update the database
        if datetime.now() - last_update_time >= timedelta(minutes=1):
            last_update_time = datetime.now()  # Reset the timer
            # Get current date and time
            now = datetime.now()
            date = now.strftime("%d-%m-%Y")
            time = now.strftime("%H:%M")

            # Insert data into the MongoDB collection
            collection.insert_one({
                'date': date,
                'time': time,
                'enter': count_enter,
                'exit': count_exit,
                'total': count_enter + count_exit
            })

        cv2.imshow(f"RGB {cam_index}", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        await asyncio.sleep(0.01)  # Yield control to allow other tasks to run

    cap.release()
    cv2.destroyAllWindows()

# Main function to start WebSocket server and video processing


async def main():
    video_tasks = [asyncio.create_task(process_video(i))
                   for i in range(len(caps))]
    await asyncio.gather(*video_tasks)

# Run the main function
asyncio.run(main())
