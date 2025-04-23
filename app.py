import streamlit as st
import tempfile
import cv2
import pandas as pd
from collections import defaultdict
from ultralytics import YOLO
import os

# Constants
VEHICLE_MODEL_PATH = "models/typess.pt"
CONGESTION_MODEL_PATH = "models/vehcongestion.pt"
EXCEL_FILE = "detection_log.xlsx"
CONF_THRESHOLD = 0.5
IMG_SIZE = 640
FRAME_SKIP = 1  # Process every frame

# Load YOLO models
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
congestion_model = YOLO(CONGESTION_MODEL_PATH)
vehicle_class_names = vehicle_model.names

st.title("📹 Video Vehicle Detection + Congestion Logging")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    # Prepare log
    if os.path.exists(EXCEL_FILE):
        df_log = pd.read_excel(EXCEL_FILE)
    else:
        df_log = pd.DataFrame(columns=["Frame", "Total_Vehicles", "Car", "Truck", "Bus", "Other", "Congestion_Status"])

    stframe = st.empty()
    frame_num = 0

    st.write("⏳ Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % FRAME_SKIP != 0:
            frame_num += 1
            continue

        vehicle_counts = defaultdict(int)
        v_results = vehicle_model.predict(frame, verbose=False)[0]

        for box in v_results.boxes:
            cls_id = int(box.cls)
            label = vehicle_class_names[cls_id]
            vehicle_counts[label] += 1

        car_count = vehicle_counts.get("car", 0)
        truck_count = vehicle_counts.get("truck", 0)
        bus_count = vehicle_counts.get("bus", 0)
        total_vehicles = sum(vehicle_counts.values())
        other_count = total_vehicles - car_count - truck_count - bus_count

        c_results = congestion_model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        congestion_status = "Non-Congested"
        for box in c_results.boxes:
            if int(box.cls) == 2:  # Congested
                congestion_status = "Congested"
                break

        annotated = v_results.plot()
        cv2.putText(annotated, f"Total: {total_vehicles} | {congestion_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        stframe.image(annotated, channels="BGR", caption=f"Frame {frame_num}", use_column_width=True)

        # Log to DataFrame
        df_log.loc[len(df_log)] = {
            "Frame": frame_num,
            "Total_Vehicles": total_vehicles,
            "Car": car_count,
            "Truck": truck_count,
            "Bus": bus_count,
            "Other": other_count,
            "Congestion_Status": congestion_status
        }

        frame_num += 1

    cap.release()
    df_log.to_excel(EXCEL_FILE, index=False)
    st.success("✅ Detection complete and logged.")
    st.download_button("📥 Download Detection Log", data=df_log.to_csv(index=False),
                        file_name="detection_log.csv", mime="text/csv")
