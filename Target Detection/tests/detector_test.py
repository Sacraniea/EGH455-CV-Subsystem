# tests/test_predict.py

import cv2
import pytest

from src.predict import initialise_predictor

CLASSES = ("gauge", "openValve", "closedValve")
MAX_COUNT_PER_TYPE = 10
MAX_ATTEMPTS = 100


def show(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


@pytest.fixture(scope="module")
def predictor():
    return initialise_predictor()


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


class TestPredictor:
    def general_test(self, predictor, samples_by_class, class_type):
        samples = samples_by_class[class_type]
        if not samples:
            pytest.skip(f"No {class_type} samples found in {MAX_ATTEMPTS} draws")

        for raw_image, ground_labels in samples:
            prediction = predictor.predict(raw_image)
            predicted_image = predictor.annotate_image(raw_image, prediction)

            ground_classes = predictor.extract_only_labels(ground_labels)
            predicted_classes = predictor.extract_only_labels(prediction)
            print(f"Predicted: {predicted_classes}, Ground: {ground_classes}")

            show("Raw", raw_image)
            show(f"{class_type} Prediction", predicted_image)
            assert ground_classes == predicted_classes

    def test_spam_gauge(self, predictor, samples_by_class):
        self.general_test(predictor, samples_by_class, "gauge")

    def test_spam_openvalve(self, predictor, samples_by_class):
        self.general_test(predictor, samples_by_class, "openValve")

    def test_spam_closedvalve(self, predictor, samples_by_class):
        self.general_test(predictor, samples_by_class, "closedValve")