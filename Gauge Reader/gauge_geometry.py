"""Convert gauge points, angles, and calibrated scale positions."""

import math
from typing import Optional

from gauge_types import GaugeCalibration, Point


def normalise_angle_degrees(angle_degrees: float) -> float:
    """Wrap an angle into the half-open interval [0, 360) degrees.

    Args:
        angle_degrees: An angle that may be negative or exceed one rotation.
    Returns:
        The normalised angle in the range [0, 360) degrees.
    """
    return angle_degrees % 360.0


def angle_from_points(centre: Point, tip: Point) -> float:
    """Calculate the clockwise clock-face angle from centre to the tip of the needle.

    Args:
        centre: The (x, y) coordinates of the gauge's centre.
        tip: The (x, y) coordinates of the needle's tip.

    Returns:
        The clockwise clock-face angle in the range [0, 360) degrees.

    Raises:
        ValueError: If the centre and tip are identical, as this would
        produce a zero-length needle with no direction.
    """

    dx = tip[0] - centre[0]
    dy = tip[1] - centre[1]

    if dx == 0.0 and dy == 0.0:
        raise ValueError("Centre and tip cannot be identical.")

    # Calculate the needle direction with up as zero and clockwise as positive.
    angle_radians = math.atan2(dx, -dy)
    angle_degrees = math.degrees(angle_radians)

    return normalise_angle_degrees(angle_degrees)


def point_at_angle(
    centre: Point,
    distance: float,
    angle_degrees: float,
) -> Point:
    """Constructs a point at a clock-face angle and distance from the centre.

    This is the inverse operation of ``angle_from_points``.

    Args:
        centre: The (x, y) coordinates of the gauge's centre.
        distance: The distance from the centre to the point in pixels.
        angle_degrees: The clockwise angle measured from the upwards direction.


    Returns:
        The (x, y) coordinates of the point at the specified angle and distance.
    """

    angle_radians = math.radians(normalise_angle_degrees(angle_degrees))

    # Add the horizontal distance from the centre to the requested point.
    x = centre[0] + distance * math.sin(angle_radians)

    # Subtract the vertical distance so a zero-degree point sits above the centre.
    y = centre[1] - distance * math.cos(angle_radians)

    return (x, y)


def pressure_from_angle(
    angle_degrees: float,
    calibration: GaugeCalibration,
) -> Optional[float]:
    """Convert an absolute needle angle into a pressure reading.

    Args:
        angle_degrees: Absolute clockwise clock-face angle of the needle.
        calibration: Definition of the gauge range and valid scale sweep.

    Returns:
        Pressure in bar, using the nearest endpoint for a small overshoot.
        Angles farther outside the calibrated scale return None.

    """

    # Measure how far the needle has moved clockwise from the gauge's zero mark.
    travel_degrees = normalise_angle_degrees(
        angle_degrees - calibration.zero_angle_degrees
    )

    # Allow a small angle error beyond either end without shifting the scale.
    if travel_degrees > calibration.clockwise_sweep_degrees:
        below_zero_degrees = 360.0 - travel_degrees
        above_full_scale_degrees = travel_degrees - calibration.clockwise_sweep_degrees
        tolerance = calibration.endpoint_tolerance_degrees

        # A pointer just below zero reads the minimum pressure, never a negative value.
        if (
            below_zero_degrees <= tolerance
            and below_zero_degrees < above_full_scale_degrees
        ):
            return calibration.minimum_bar

        # A pointer just beyond full scale reads the maximum pressure.
        if (
            above_full_scale_degrees <= tolerance
            and above_full_scale_degrees < below_zero_degrees
        ):
            return calibration.maximum_bar

        # Keep withholding pressure when the angle is too far outside or ambiguous.
        return None

    # Divide the angle travelled by the full sweep to get the fraction of the scale.
    progress = travel_degrees / calibration.clockwise_sweep_degrees

    # Apply that fraction between the configured minimum and maximum pressure in bar.
    pressure_range = calibration.maximum_bar - calibration.minimum_bar
    return calibration.minimum_bar + progress * pressure_range
