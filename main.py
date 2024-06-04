import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone

# Load YOLO model
model = YOLO('yolov8s.pt')


# Function to capture mouse events (currently just printing coordinates)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


# Set up mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load video
cap = cv2.VideoCapture('vidp.mp4')

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
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
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
            person_down[id] = (cx, cy)
        if id in person_down and (cy + offset) > cy2 > (cy - offset):
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
            if id not in counter_down:
                counter_down.append(id)

        # Upside Counter
        if (cy + offset) > cy2 > (cy - offset):
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
            person_up[id] = (cx, cy)
        if id in person_up and (cy + offset) > cy1 > (cy - offset):
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
            cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
            if id not in counter_up:
                counter_up.append(id)

    # Draw lines and counters
    cv2.line(frame, (3, cy1), (1018, cy1), (0, 255, 0), 2)
    cv2.line(frame, (5, cy2), (1019, cy2), (0, 255, 255), 2)

    down = len(counter_down)
    up = len(counter_up)
    cvzone.putTextRect(frame, f'Enter: {down}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'Exit: {up}', (50, 100), 2, 2)
    cvzone.putTextRect(frame, f'Total: {up + down}', (800, 60), 2, 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
