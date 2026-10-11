import unittest

import cv2
import numpy as np

from aruco_types import ArUcoConfig
from aruco_detection import detect_markers


class ArUcoDetectionTests(unittest.TestCase):

    def test_identifies_generated_marker(self):
        # Place marker 7 at known pixel coordinates and compare the detected ID and corners.
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        marker = cv2.aruco.generateImageMarker(dictionary, 7, 200)

        image = np.full((300, 300), 255, dtype=np.uint8)
        image[50:250, 50:250] = marker
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({7}),
        )

        observations = detect_markers(image, config)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].marker_id, 7)

        # The 200-pixel marker occupies coordinates 50 through 249 on each axis.
        expected_corners = (
            (50.0, 50.0),
            (249.0, 50.0),
            (249.0, 249.0),
            (50.0, 249.0),
        )

        np.testing.assert_allclose(
            observations[0].corners_px,
            expected_corners,
            rtol=0,
            atol=0.5,
        )

    def test_returns_no_observations_for_blank_image(self):
        image = np.full((300, 300, 3), 255, dtype=np.uint8)

        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({7}),
        )

        observations = detect_markers(image, config)

        self.assertEqual(observations, ())

    def test_does_not_modify_source_image(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        marker = cv2.aruco.generateImageMarker(dictionary, 7, 200)

        image = np.full((300, 300), 255, dtype=np.uint8)
        image[50:250, 50:250] = marker
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        original = image.copy()

        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({7}),
        )

        detect_markers(image, config)

        np.testing.assert_array_equal(image, original)

    def test_returns_multiple_markers_in_id_order(self):
        # Detect two markers in one image and require results sorted by their IDs.
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

        image = np.full((300, 550), 255, dtype=np.uint8)
        image[50:250, 50:250] = cv2.aruco.generateImageMarker(dictionary, 7, 200)
        image[50:250, 300:500] = cv2.aruco.generateImageMarker(dictionary, 23, 200)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({7, 23}),
        )

        observations = detect_markers(image, config)

        self.assertEqual(
            [observation.marker_id for observation in observations],
            [7, 23],
        )

    def test_rejects_missing_image(self):
        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({7}),
        )

        with self.assertRaises(ValueError):
            detect_markers(None, config)


if __name__ == "__main__":
    unittest.main()
