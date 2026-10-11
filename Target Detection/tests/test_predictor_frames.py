import numpy as np
import pytest
from ultralytics.engine.results import Results

from src import predict as predictor_module


DETECTION_ROWS = [
    [32, 16, 96, 80, 0.95, 1],
    [128, 32, 224, 112, 0.90, 1],
    [208, 0, 256, 32, 0.85, 2],
]

EXPECTED_LABELS = [
    ("gauge", (0.25, 0.375, 0.25, 0.5)),
    ("gauge", (0.6875, 0.5625, 0.375, 0.625)),
    ("openValve", (0.90625, 0.125, 0.1875, 0.25)),
]


@pytest.fixture
def image():
    # Give each pixel its column and row so crop offsets can be checked exactly.
    rows, columns = np.indices((128, 256))
    return np.stack((columns, rows, (columns + rows) % 256), axis=-1).astype(np.uint8)


@pytest.fixture
def make_predictor(monkeypatch):
    def construct(frames):
        # Substitute only model loading and inference; mapping and cropping stay real.
        pending = iter(frames)

        class ScriptedModel:
            def predict(self, image, verbose=False):
                boxes = np.asarray(next(pending), dtype=np.float32).reshape(-1, 6)
                return [
                    Results(
                        orig_img=image,
                        path="fixture.png",
                        names={0: "closedValve", 1: "gauge", 2: "openValve"},
                        boxes=boxes,
                    )
                ]

        monkeypatch.setattr(predictor_module.Predictor, "_instance", None)
        monkeypatch.setattr(predictor_module, "YOLO", lambda path: ScriptedModel())
        return predictor_module.Predictor("unused-test-weights.pt")

    return construct


def test_returns_each_current_detection_with_its_own_box_and_gauge_crop(
    make_predictor, image
):
    predictor = make_predictor([DETECTION_ROWS])

    detections = predictor.predict_detections(image)

    assert [detection["class_name"] for detection in detections] == [
        "gauge",
        "gauge",
        "openValve",
    ]
    assert [tuple(detection["box_xywhn"]) for detection in detections] == [
        label[1] for label in EXPECTED_LABELS
    ]
    assert [tuple(detection["bbox_xyxy_px"]) for detection in detections] == [
        (32, 16, 96, 80),
        (128, 32, 224, 112),
        (208, 0, 256, 32),
    ]

    # Each gauge gets pixels from its own box; the valve has no gauge crop.
    np.testing.assert_array_equal(detections[0]["gauge_crop"], image[16:80, 32:96])
    np.testing.assert_array_equal(detections[1]["gauge_crop"], image[32:112, 128:224])
    assert detections[2]["gauge_crop"] is None


def test_gauge_crops_are_clean_independent_images_and_source_is_preserved(
    make_predictor, image
):
    predictor = make_predictor([DETECTION_ROWS])
    original = image.copy()
    image.flags.writeable = False

    detections = predictor.predict_detections(image)
    first_crop = detections[0]["gauge_crop"]
    second_crop = detections[1]["gauge_crop"]

    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(first_crop, original[16:80, 32:96])
    np.testing.assert_array_equal(second_crop, original[32:112, 128:224])
    assert not np.shares_memory(first_crop, image)
    assert not np.shares_memory(second_crop, image)
    assert not np.shares_memory(first_crop, second_crop)

    # Editing one returned crop must leave the source and the other gauge untouched.
    first_crop[:] = 0
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(second_crop, original[32:112, 128:224])


def test_latest_gauge_crop_is_unavailable_before_first_prediction(make_predictor):
    predictor = make_predictor([])

    assert predictor.get_latest_gauge_crop() is None


@pytest.mark.parametrize("method_name", ["predict", "predict_detections"])
def test_empty_frame_clears_previous_gauge_crop(make_predictor, image, method_name):
    predictor = make_predictor([DETECTION_ROWS, []])
    predict = getattr(predictor, method_name)
    predict(image)
    np.testing.assert_array_equal(
        predictor.get_latest_gauge_crop(), image[32:112, 128:224]
    )

    # An empty new frame must not expose the gauge retained from the previous frame.
    assert predict(np.zeros_like(image)) == []
    assert predictor.get_latest_gauge_crop() is None


def test_predict_keeps_normalized_labels_for_existing_annotation_code(
    make_predictor, image
):
    predictor = make_predictor([DETECTION_ROWS])

    labels = predictor.predict(image)

    assert labels == EXPECTED_LABELS
    annotated = predictor.annotate_image(image, labels)
    assert annotated.shape == image.shape
    assert np.any(annotated != image)
