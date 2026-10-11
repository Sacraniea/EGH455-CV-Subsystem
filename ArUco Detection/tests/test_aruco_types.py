import unittest

from aruco_types import ArUcoConfig, MarkerObservation, CameraCalibration


class ArUcoConfigTests(unittest.TestCase):

    def test_rejects_non_positive_marker_size(self):
        for marker_size in (0.0, -0.2):
            with self.subTest(marker_size=marker_size):
                with self.assertRaises(ValueError):
                    ArUcoConfig(
                        marker_side_length_m=marker_size,
                        dictionary_name="DICT_5X5_100",
                        permitted_ids=frozenset({0}),
                    )

    def test_accepts_positive_marker_size(self):
        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({0}),
        )

        self.assertEqual(config.marker_side_length_m, 0.2)

    def test_rejects_non_finite_marker_size(self):
        for marker_size in (
            float("inf"),
            float("-inf"),
            float("nan"),
        ):
            with self.subTest(marker_size=marker_size):
                with self.assertRaises(ValueError):
                    ArUcoConfig(
                        marker_side_length_m=marker_size,
                        dictionary_name="DICT_5X5_100",
                        permitted_ids=frozenset({0}),
                    )

    def test_rejects_unkown_dictionary(self):
        with self.assertRaises(ValueError):
            ArUcoConfig(
                marker_side_length_m=0.2,
                dictionary_name="NOT_A_DICTIONARY",
                permitted_ids=frozenset({0}),
            )

    def test_rejects_out_of_range_permitted_ids(self):
        for marker_id in (-1, 100):
            with self.subTest(marker_id=marker_id):
                with self.assertRaises(ValueError):
                    ArUcoConfig(
                        marker_side_length_m=0.2,
                        dictionary_name="DICT_5X5_100",
                        permitted_ids=frozenset({marker_id}),
                    )

    def test_accepts_valid_dictuinary_boundary_ids(self):
        config = ArUcoConfig(
            marker_side_length_m=0.2,
            dictionary_name="DICT_5X5_100",
            permitted_ids=frozenset({0, 99}),
        )

        self.assertEqual(config.permitted_ids, frozenset({0, 99}))

    def test_rejects_non_integer_permitted_ids(self):
        # Verify that fractional, text and boolean marker IDs all raise TypeError.
        for marker_id in (1.5, "7", True):
            with self.subTest(marker_id=marker_id):
                with self.assertRaises(TypeError):
                    ArUcoConfig(
                        marker_side_length_m=0.2,
                        dictionary_name="DICT_5X5_100",
                        permitted_ids=frozenset({marker_id}),
                    )


class MarkerObservationTests(unittest.TestCase):
    def test_rejects_missing_marker_corner(self):
        with self.assertRaises(ValueError):
            MarkerObservation(
                marker_id=7,
                corners_px=(
                    (10.0, 10.0),
                    (30.0, 10.0),
                    (30.0, 30.0),
                ),
            )

    def test_accepts_four_corners_in_detector_order(self):
        # Start at a different corner and check that construction preserves this order.
        corners = (
            (30.0, 10.0),
            (30.0, 30.0),
            (10.0, 30.0),
            (10.0, 10.0),
        )

        observation = MarkerObservation(
            marker_id=7,
            corners_px=corners,
        )

        self.assertEqual(observation.marker_id, 7)
        self.assertEqual(observation.corners_px, corners)

    def test_rejects_non_finite_corner_coordinates(self):
        for invalid_value in (
            float("inf"),
            float("-inf"),
            float("nan"),
        ):
            for axis in (0, 1):
                # Replace x and y separately so either invalid coordinate is rejected.
                with self.subTest(value=invalid_value, axis=axis):
                    corner = [10.0, 10.0]
                    corner[axis] = invalid_value

                    with self.assertRaises(ValueError):
                        MarkerObservation(
                            marker_id=7,
                            corners_px=(
                                (10.0, 10.0),
                                (30.0, 30.0),
                                (10.0, 30.0),
                                tuple(corner),
                            ),
                        )


class CameraCalibrationTests(unittest.TestCase):
    def test_rejects_camera_matrix_with_missing_row(self):
        with self.assertRaises(ValueError):
            CameraCalibration(
                camera_matrix=(
                    (1000.0, 0.0, 960.0),
                    (0.0, 1000.0, 540.0),
                ),
                distortion_coefficients=(0.0,) * 5,
                image_size_px=(1920, 1080),
                source="synthetic test fixture",
            )

    def test_rejects_camera_matrix_with_incomplete_row(self):
        with self.assertRaises(ValueError):
            CameraCalibration(
                camera_matrix=(
                    (1000.0, 0.0, 960.0),
                    (0.0, 1000.0),
                    (0.0, 0.0, 1.0),
                ),
                distortion_coefficients=(0.0,) * 5,
                image_size_px=(1920, 1080),
                source="synthetic test fixture",
            )

    def test_rejects_non_finite_camera_matrix_values(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    CameraCalibration(
                        camera_matrix=(
                            (value, 0.0, 960.0),
                            (0.0, 1000.0, 540.0),
                            (0.0, 0.0, 1.0),
                        ),
                        distortion_coefficients=(0.0,) * 5,
                        image_size_px=(1920, 1080),
                        source="synthetic test fixture",
                    )

    def test_rejects_non_positive_focal_lengths(self):
        # Set each focal length to zero or a negative value while the other stays valid.
        for fx, fy in (
            (0.0, 1000.0),
            (-1000.0, 1000.0),
            (1000.0, 0.0),
            (1000.0, -1000.0),
        ):
            with self.subTest(fx=fx, fy=fy):
                with self.assertRaises(ValueError):
                    CameraCalibration(
                        camera_matrix=(
                            (fx, 0.0, 960.0),
                            (0.0, fy, 540.0),
                            (0.0, 0.0, 1.0),
                        ),
                        distortion_coefficients=(0.0,) * 5,
                        image_size_px=(1920, 1080),
                        source="synthetic test fixture",
                    )

    def test_accepts_valid_calibration_and_preserves_values(self):
        matrix = (
            (1000.0, 0.0, 960.0),
            (0.0, 1000.0, 540.0),
            (0.0, 0.0, 1.0),
        )
        distortion = (0.1, -0.05, 0.0, 0.0, 0.0)

        calibration = CameraCalibration(
            camera_matrix=matrix,
            distortion_coefficients=distortion,
            image_size_px=(1920, 1080),
            source="synthetic test fixture",
        )

        self.assertEqual(calibration.camera_matrix, matrix)
        self.assertEqual(calibration.distortion_coefficients, distortion)
        self.assertEqual(calibration.image_size_px, (1920, 1080))
        self.assertEqual(calibration.source, "synthetic test fixture")

    def test_rejects_non_positive_image_dimensions(self):
        for size in (
            (0, 1080),
            (-1920, 1080),
            (1920, 0),
            (1920, -1080),
        ):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    CameraCalibration(
                        camera_matrix=(
                            (1000.0, 0.0, 960.0),
                            (0.0, 1000.0, 540.0),
                            (0.0, 0.0, 1.0),
                        ),
                        distortion_coefficients=(0.0,) * 5,
                        image_size_px=size,
                        source="synthetic test fixture",
                    )

    def test_rejects_non_finite_distortion_coefficients(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    CameraCalibration(
                        camera_matrix=(
                            (1000.0, 0.0, 960.0),
                            (0.0, 1000.0, 540.0),
                            (0.0, 0.0, 1.0),
                        ),
                        distortion_coefficients=(value, 0.0, 0.0, 0.0, 0.0),
                        image_size_px=(1920, 1080),
                        source="synthetic test fixture",
                    )


if __name__ == "__main__":
    unittest.main()
