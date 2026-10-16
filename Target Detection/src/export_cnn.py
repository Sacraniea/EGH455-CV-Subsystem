from ultralytics import YOLO
YOLO("models/50_epoch_best.pt").export(format="ncnn", imgsz=640)