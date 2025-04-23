from ultralytics import YOLO
from pathlib import Path


def test_vehicle_type_model_loading():
    model_path = Path("models/typess.pt")
    assert model_path.exists(), "Vehicle type model file not found."
    model = YOLO(str(model_path))
    assert model is not None, "Vehicle type model loading failed."

def test_congestion_model_loading():
    model_path = Path("models/vehcongestion.pt")
    assert model_path.exists(), "Congestion model file not found."
    model = YOLO(str(model_path))
    assert model is not None, "Congestion model loading failed."