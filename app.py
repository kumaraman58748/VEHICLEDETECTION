import streamlit as st
import tempfile
import cv2
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import defaultdict

# Constants
VEHICLE_MODEL_PATH = "models/typess.pt"
CONGESTION_MODEL_PATH = "models/vehcongestion.pt"
CONF_THRESHOLD = 0.53
IMG_SIZE = 640
FRAME_SKIP = 1

# Load models
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
congestion_model = YOLO(CONGESTION_MODEL_PATH)
vehicle_class_names = vehicle_model.names

st.title("🚦 Real-Time Vehicle & Congestion Detection")

mode = st.radio("Choose mode:", ["Select Mode", "Video", "Image"])

# Clear cache on mode switch
if "last_mode" not in st.session_state:
    st.session_state.last_mode = mode
if st.session_state.last_mode != mode:
    st.cache_data.clear()
    st.session_state.last_mode = mode

# Colors
light_blue = (255, 255, 102)
label_bg = (255, 255, 255)
label_shadow = (0, 0, 0)

# --- IMAGE MODE ---
if mode == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        img_np = np.array(image)

        v_results = vehicle_model(img_np, verbose=False)[0]
        c_results = congestion_model(img_np, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]

        vehicle_counts = defaultdict(int)
        for box in v_results.boxes:
            label = vehicle_class_names[int(box.cls)]
            vehicle_counts[label] += 1
        total_vehicles = sum(vehicle_counts.values())

        congestion_status = "Congested" if any(int(box.cls) == 2 for box in c_results.boxes) else "Non-Congested"

        annotated = img_np.copy()
        for box in v_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = vehicle_class_names[int(box.cls)]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), light_blue, 2)
            
            # Shadow and label
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, label_shadow, 3, lineType=cv2.LINE_AA)
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, label_bg, 1, lineType=cv2.LINE_AA)

        cv2.putText(annotated, f"Total: {total_vehicles} | {congestion_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.rectangle(annotated, (0, 0), (annotated.shape[1] - 1, annotated.shape[0] - 1), (255, 255, 153), 4)

        st.image(annotated, channels="BGR", caption="Detection Results", use_container_width=True)

        st.write("### Detection Summary")
        for cls in sorted(vehicle_counts):
            st.write(f"- {cls.capitalize()}: {vehicle_counts[cls]}")
        st.write(f"- Total Vehicles: {total_vehicles}")
        st.write(f"- Congestion: **{congestion_status}**")

# --- VIDEO MODE ---
elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        log_data = []
        frame_id = 0

        st.write("📹 Running detection in real-time...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % FRAME_SKIP != 0:
                frame_id += 1
                continue

            v_results = vehicle_model(frame, verbose=False)[0]
            c_results = congestion_model(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]

            vehicle_counts = defaultdict(int)
            for box in v_results.boxes:
                label = vehicle_class_names[int(box.cls)]
                vehicle_counts[label] += 1
            total_vehicles = sum(vehicle_counts.values())

            congestion_status = "Congested" if any(int(box.cls) == 2 for box in c_results.boxes) else "Non-Congested"

            annotated = frame.copy()
            for box in v_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = vehicle_class_names[int(box.cls)]
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), light_blue, 2)

                # Label with shadow
                cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, label_shadow, 3, lineType=cv2.LINE_AA)
                cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, label_bg, 1, lineType=cv2.LINE_AA)

            cv2.putText(annotated, f"Total: {total_vehicles} | {congestion_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1] - 1, annotated.shape[0] - 1), (255, 255, 153), 4)

            stframe.image(annotated, channels="BGR", use_container_width=True)

            log_entry = {"Frame": frame_id, "Total_Vehicles": total_vehicles, "Congestion_Status": congestion_status}
            for cls, count in vehicle_counts.items():
                log_entry[cls.capitalize()] = count
            log_data.append(log_entry)

            frame_id += 1

        cap.release()

        df_log = pd.DataFrame(log_data)
        st.success("✅ Video processing complete.")
        st.write("### Detection Log")
        st.dataframe(df_log)

        st.download_button("📥 Download Log CSV",
                           data=df_log.to_csv(index=False),
                           file_name="detection_log.csv",
                           mime="text/csv")

# --- DEFAULT MODE ---
else:
    st.info("👈 Select a mode to begin.")
