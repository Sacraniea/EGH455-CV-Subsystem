# Run with pixi run python src/export_cnn.py (from Target Detection directory)
from ultralytics import YOLO
YOLO("models/v4.pt").export(format="ncnn", imgsz=640)