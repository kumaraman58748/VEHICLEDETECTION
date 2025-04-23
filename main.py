import cv2
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
import os

VEHICLE_MODEL_PATH = 'models/typess.pt'
CONGESTION_MODEL_PATH = 'models/vehcongestion.pt'
VIDEO_SOURCE = 'videos/congested.mp4'
EXCEL_FILE = 'detection_log.xlsx'
CONF_THRESHOLD = 0.5
IMG_SIZE = 640
FRAME_SKIP = 1

vehicle_model = YOLO(VEHICLE_MODEL_PATH)
congestion_model = YOLO(CONGESTION_MODEL_PATH)
vehicle_class_names = vehicle_model.names

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Could not open video source.")
    exit()

if os.path.exists(EXCEL_FILE):
    df_log = pd.read_excel(EXCEL_FILE)
else:
    df_log = pd.DataFrame(columns=["Frame", "Total_Vehicles", "Car", "Truck", "Bus", "Other", "Congestion_Status"])

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_num % FRAME_SKIP != 0:
        frame_num += 1
        continue

    vehicle_counts = defaultdict(int)
    v_results = vehicle_model(frame, verbose=False)[0]
    for box in v_results.boxes:
        cls_id = int(box.cls)
        label = vehicle_class_names[cls_id]
        vehicle_counts[label] += 1

    car_count = vehicle_counts.get("car", 0)
    truck_count = vehicle_counts.get("truck", 0)
    bus_count = vehicle_counts.get("bus", 0)
    total_vehicles = sum(vehicle_counts.values())
    other_count = total_vehicles - car_count - truck_count - bus_count

    c_results = congestion_model(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)[0]
    congestion_status = "Non-Congested"
    for box in c_results.boxes:
        cls_id = int(box.cls)
        if cls_id == 2:  # Class 2 = Congested
            congestion_status = "Congested"
            break

    annotated_frame = v_results.plot()
    cv2.putText(annotated_frame, f"Total: {total_vehicles} | Status: {congestion_status}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Live Detection", annotated_frame)

    df_log.loc[len(df_log)] = {
        "Frame": frame_num,
        "Total_Vehicles": total_vehicles,
        "Car": car_count,
        "Truck": truck_count,
        "Bus": bus_count,
        "Other": other_count,
        "Congestion_Status": congestion_status
    }

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1

cap.release()
df_log.to_excel(EXCEL_FILE, index=False)
cv2.destroyAllWindows()
print(f"Detection complete. Data logged to {EXCEL_FILE}")
