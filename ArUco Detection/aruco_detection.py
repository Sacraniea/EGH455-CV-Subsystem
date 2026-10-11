import cv2
import numpy as np

from aruco_types import ArUcoConfig, MarkerObservation


def detect_markers(
    image: np.ndarray,
    config: ArUcoConfig,
) -> tuple[MarkerObservation, ...]:
    # Detect marker IDs and four-corner positions in a colour frame.
    # Reject empty images or arrays that are not uint8 BGR pixels.
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.size == 0
    ):
        raise ValueError("Input image must be a non-empty uint8 BGR array (H, W, 3)")

    # Create the detector using the marker dictionary selected in the configuration.
    dictionary_id = getattr(cv2.aruco, config.dictionary_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    detector = cv2.aruco.ArucoDetector(dictionary)

    # Detect on a grayscale copy and keep the source colour image unchanged.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)

    # Return no observations when OpenCV did not decode any marker IDs.
    if ids is None:
        return ()

    observations = []

    for marker_id, marker_corners in zip(ids.flatten(), corners):
        # Reshape this marker into four (x, y) points without changing their order.
        points = marker_corners.reshape(4, 2)

        # Store ordinary numeric values instead of OpenCV's array elements.
        observation = MarkerObservation(
            marker_id=int(marker_id),
            corners_px=tuple((float(x), float(y)) for x, y in points),
        )
        observations.append(observation)

    # Sort by marker ID, then by corner coordinates when IDs are repeated.
    return tuple(
        sorted(
            observations,
            key=lambda observation: (
                observation.marker_id,
                observation.corners_px,
            ),
        )
    )
