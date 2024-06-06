import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone
import asyncio
import websockets

# Load YOLO model
model = YOLO('yolov8s.pt')

# Function to capture mouse events (just printing coordinates)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


# Set up mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

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
cy1, cy2, offset = 194, 220, 6

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

        results = model.predict(frame)
        a = results[0].boxes.data
        # The line `px = pd.DataFrame(a).astype("float")` is creating a pandas DataFrame from the data
        # in `a` and then converting all the values in the DataFrame to float data type.
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
                # cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                person_down[id] = (cx, cy)
            if id in person_down and (cy + offset) > cy2 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                # cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                if id not in counter_down:
                    counter_down.append(id)

            # Upside Counter
            if (cy + offset) > cy2 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                # cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                person_up[id] = (cx, cy)
            if id in person_up and (cy + offset) > cy1 > (cy - offset):
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                # cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                if id not in counter_up:
                    counter_up.append(id)

        # Draw lines and counters
        cv2.line(frame, (3, cy1), (1018, cy1), (0, 255, 0), 2)
        cv2.line(frame, (5, cy2), (1019, cy2), (0, 255, 255), 2)

        down = len(counter_down)
        up = len(counter_up)
        shared_data = f'Enter: {down}, Exit: {up}, Total: {up + down}'
        cvzone.putTextRect(frame, f'Enter: {down}', (50, 60), 2, 2)
        cvzone.putTextRect(frame, f'Exit: {up}', (50, 100), 2, 2)
        cvzone.putTextRect(frame, f'Total: {up + down}', (800, 60), 2, 2)

       # The `cv2.waitKey(1)` function waits for a key event for 1 millisecond. 
       # This allows the user to exit the video processing loop by pressing the 'Esc' key.
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        await asyncio.sleep(0.01)  # It allow other tasks to run

    cap.release()            #used to release the video capture object
    cv2.destroyAllWindows()

# WebSocket server function
async def communication(websocket, path):
    global shared_data

    while True:
        if shared_data:
            await websocket.send(shared_data)
        await asyncio.sleep(0)  # Adjust the delay as needed

# Main function to start WebSocket server and video processing
async def main():
    start_server = websockets.serve(communication, "localhost", 8765)
    await start_server

    video_task = asyncio.create_task(process_video())
    await asyncio.Future()  # Run forever

# Run the main function
asyncio.run(main())
