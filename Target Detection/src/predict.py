"""
predict - YOLO object detection to find gauges and valves in images
"""

from ultralytics import YOLO
from annotater import Annotater
import cv2
from datetime import datetime, timedelta

from pathlib import Path


def initialise_predictor():
    MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "50_epoch_best_ncnn_model"
    model_path = Path(
        MODEL_PATH
    ).expanduser().resolve()

    print(model_path)
    return Predictor(model_path)


class Predictor(Annotater):
    """Predictor class for YOLO object detection to find gauges and valves in images."""

    _instance = None  # Class-level variable to store the single instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create the instance if it doesn't exist yet
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_location: str):
        super().__init__()
        self.model = YOLO(model_location, task="detect")
        self.latest_prediction = []
        self.time_last_gauge = None
        self.latest_gauge_image = None

    def test_the_class_works(self):
        print("\n\n\n\n\nYay you imported correctly!\n\n\n\n")
        return


    def predict(self, raw_image):
        # Keep the original label format for the existing image demos.
        detections = self.predict_detections(raw_image)
        return [
            (detection["class_name"], detection["box_xywhn"])
            for detection in detections
        ]


    def predict_detections(self, raw_image):
        """Return this frame's targets with pixel boxes and clean gauge crops."""

        # Clear the previous frame before looking for targets in this one.
        self.latest_prediction = []
        self.latest_gauge_image = None
        self.time_last_gauge = None

        # Run the model on a copy so the camera's original image stays untouched.
        image = raw_image.copy()
        boxes = self.model.predict(image, verbose=False)[0].boxes
        cleaned_prediction = [
            (self.class_names[int(cls)], tuple(xywhn))
            for cls, xywhn in zip(boxes.cls.tolist(), boxes.xywhn.tolist())
        ]

        # Prepare each crop once and return it beside the box that produced it.
        return self.__cleanup_post_prediction(cleaned_prediction, image)

    def get_latest_gauge_crop(self):
        # There is no preview crop before a gauge is found, or after an empty frame.
        if self.time_last_gauge is None:
            return None

        # Keep the existing single-gauge preview available for up to 30 seconds.
        if (datetime.now() - self.time_last_gauge) < timedelta(seconds=30):
            return self.latest_gauge_image

        else:
            return None


    def demonstrate(self, image):
        results = self.predict(image)
        annotated_image = self.annotate_image(image, results)
        cv2.imshow(annotated_image)

    ####### PRIVATE FUNCTIONS #################

    def __cleanup_post_prediction(self, cleaned_prediction, image):
        # Keep the normalized labels available for Aidan's existing annotation code.
        self.latest_prediction = cleaned_prediction
        detections = []

        for name, box in cleaned_prediction:
            # Convert this target's box once, using the original frame dimensions.
            pixel_box = self.__to_pixel_box(image, box)
            gauge_crop = None

            if name == "gauge":
                left, top, right, bottom = pixel_box

                # Give each gauge its own clean crop, before any labels are drawn.
                gauge_crop = image[top:bottom, left:right].copy()
                self.latest_gauge_image = gauge_crop
                self.time_last_gauge = datetime.now()

            # Return every target; the preview above only holds the last gauge.
            detections.append({
                "class_name": name,
                "box_xywhn": box,
                "bbox_xyxy_px": pixel_box,
                "gauge_crop": gauge_crop,
            })

        return detections


    def __to_pixel_box(self, image, box):
        # Turn the normalized centre and size into the frame's pixel edges.
        xc, yc, w, h = map(float, box)
        H, W = image.shape[:2]
        x1 = max(int((xc - w / 2) * W), 0)
        y1 = max(int((yc - h / 2) * H), 0)
        x2 = min(int((xc + w / 2) * W), W)
        y2 = min(int((yc + h / 2) * H), H)
        return x1, y1, x2, y2
