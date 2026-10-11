# tests/test_predict.py

import cv2
import pytest

from src.predict import Predictor
from gauge_reader import read_gauge

CLASSES = ("gauge", "openValve", "closedValve")
MAX_COUNT_PER_TYPE = 10
MAX_ATTEMPTS = 100


def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def side_by_side(left, right):
    """Scale `right` to left's height, then place them next to each other."""
    h = left.shape[0]
    scale = h / right.shape[0]
    right = cv2.resize(right, (int(right.shape[1] * scale), h))
    return cv2.hconcat([left, right])


@pytest.fixture(scope="module")
def predictor():
    return Predictor("models/50_epoch_best.pt")


@pytest.fixture(scope="module")
def samples_by_class(predictor):
    """One pass over random samples, bucketed by class present in the ground labels."""
    samples = {name: [] for name in CLASSES}
    for _ in range(MAX_ATTEMPTS):
        if all(len(s) >= MAX_COUNT_PER_TYPE for s in samples.values()):
            break
        raw_image, _, ground_labels = predictor.get_test_sample()
        for name in {n for n, _ in ground_labels} & samples.keys():
            if len(samples[name]) < MAX_COUNT_PER_TYPE:
                samples[name].append((raw_image, ground_labels))
    return samples


class TestGaugeCropping:
    
    def test_gauge_cropping(self, predictor, samples_by_class):
        class_type = "gauge"
        samples = samples_by_class[class_type]
        if not samples:
            pytest.skip(f"No {class_type} samples found in {MAX_ATTEMPTS} draws")

        for raw_image, ground_labels in samples:

            ## ONE RUN OF GAUGE DETECTION AND CROPPING
            prediction = predictor.predict(raw_image) # [("gauge", (x1, y1, x2, y2)), ... ]
            predicted_image = predictor.annotate_image(raw_image, prediction) # predicted_image =Image with bounding box

            ground_classes = predictor.extract_only_labels(ground_labels) # just transforms [("gauge", (x1, y1, x2, y2)), ("openValve", (x1, y1, x2, y2))] -> ["gauge", "openValve"]
            predicted_classes = predictor.extract_only_labels(prediction)

            gauge_crop_image = predictor.get_latest_gauge_crop() # a pure image of just the gauge face (cropped based off of the detection)
            
            if gauge_crop_image is not None:

                # Read the cropped gauge image and call gauge detection
                result = read_gauge(gauge_crop_image)

                # Store the results in a readable dict for printing/logging
                reading_result = result.to_dict()
                print(reading_result)

                split_view = side_by_side(raw_image, gauge_crop_image)
                show("Raw Gauge vs Cropped Prediction", split_view)
            else:
                show("No gauge detected :/", raw_image)
            assert 1 + 1 == 2

        