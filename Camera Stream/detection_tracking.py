"""Match targets between frames and confirm pressure from agreeing readings."""

from collections import Counter, deque
from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True)
class TrackingConfig:
    # Require this overlap-to-combined-area ratio before matching an earlier target.
    minimum_iou: float = 0.3

    # Forget a target when it has not been seen for longer than this many seconds.
    maximum_gap_s: float = 2.0

    # Require this many consecutive readings before showing a pressure.
    confirmation_frames: int = 3

    # Limit the difference between the highest and lowest readings in that group.
    tolerance_bar: float = 0.2


@dataclass
class TargetTrack:
    """Keep one target's last position and recent pressure readings."""

    track_id: str
    group: str
    bbox: tuple
    last_seen_s: float
    candidates: deque = field(default_factory=deque)


def intersection_over_union(first, second):
    """Compare two boxes using their shared area divided by their combined area."""

    # Boxes use left, top, right and bottom coordinates.
    # Find the rectangle covered by both boxes, or zero area if they do not overlap.
    overlap_left = max(first[0], second[0])
    overlap_top = max(first[1], second[1])
    overlap_right = min(first[2], second[2])
    overlap_bottom = min(first[3], second[3])

    width = max(0, overlap_right - overlap_left)
    height = max(0, overlap_bottom - overlap_top)
    overlap = width * height

    # Add both box areas and subtract the overlap so it is only counted once.
    first_area = (
        max(0, first[2] - first[0])
        * max(0, first[3] - first[1])
    )
    second_area = (
        max(0, second[2] - second[0])
        * max(0, second[3] - second[1])
    )
    union = first_area + second_area - overlap

    if union > 0:
        return overlap / union

    # Boxes with no combined area cannot provide an overlap match.
    return 0.0


class TargetTracker:
    def __init__(self, config = TrackingConfig()):
        self.config = config
        self._next_id = 1
        self.reset()

    def reset(self):
        # Discard target positions and pressure histories after a processing break.
        # Leave _next_id unchanged so new targets cannot reuse earlier IDs.
        self._tracks = {}
        self._previous_identity = None

    def associate(self, predictions, identity):
        """Return one track for each prediction, in the same order as the predictions."""

        timestamp = identity.device_timestamp_s
        previous = self._previous_identity

        # Stop tracking if the frame timestamp is not a usable number.
        if not isfinite(timestamp):
            self.reset()
            raise ValueError("Target tracking requires a finite capture timestamp")

        # Restart matching if the sequence or time repeats or moves backwards,
        # or if the camera image changes size.
        if (
            previous is not None
            and (
                identity.sequence_number <= previous.sequence_number
                or timestamp <= previous.device_timestamp_s
                or identity.image_size_px != previous.image_size_px
            )
        ):
            self.reset()

        self._previous_identity = identity

        # Retain only targets seen recently enough to match this frame.
        recent_tracks = {}

        for track_id, track in self._tracks.items():
            if timestamp - track.last_seen_s <= self.config.maximum_gap_s:
                recent_tracks[track_id] = track

        self._tracks = recent_tracks

        # Keep a valve's identity when its detected state changes from open to closed.
        # Gauges and other target classes remain separate matching groups.
        groups = []

        for prediction in predictions:
            class_name = prediction["class_name"]

            if class_name in ("openValve", "closedValve"):
                groups.append("valve")

            else:
                groups.append(class_name)

        # Collect every earlier target that could match each current detection.
        candidate_matches = []

        for prediction, group in zip(predictions, groups):
            candidate_ids = []

            for track_id, track in self._tracks.items():
                if track.group != group:
                    continue

                overlap = intersection_over_union(
                    track.bbox,
                    prediction["bbox_xyxy_px"],
                )

                if overlap >= self.config.minimum_iou:
                    candidate_ids.append(track_id)

            candidate_matches.append(candidate_ids)

        # Count how many current detections could claim each earlier target.
        match_counts = Counter()

        for candidate_ids in candidate_matches:
            match_counts.update(candidate_ids)

        # Reuse a history only when there is exactly one match in both directions.
        matched_ids = []

        for candidate_ids in candidate_matches:
            if (
                len(candidate_ids) == 1
                and match_counts[candidate_ids[0]] == 1
            ):
                matched_ids.append(candidate_ids[0])

            else:
                matched_ids.append(None)

        # Forget old tracks involved in uncertain matches. Keeping them could lend
        # one gauge's confirmed pressure history to a different gauge.
        ambiguous_ids = set()

        for candidate_ids, match in zip(candidate_matches, matched_ids):
            if match is None:
                ambiguous_ids.update(candidate_ids)

        for track_id in ambiguous_ids:
            self._tracks.pop(track_id, None)

        # A missing target may keep its ID briefly, but its readings are no longer
        # consecutive. Clear that history even when the target has not expired yet.
        for track_id, track in self._tracks.items():
            if track_id not in matched_ids:
                track.candidates.clear()

        current_tracks = []

        for prediction, group, match in zip(predictions, groups, matched_ids):
            bbox = tuple(prediction["bbox_xyxy_px"])

            if match is None:
                # Start a new identity and an empty pressure history for this target.
                track = TargetTrack(
                    track_id = f"target-{self._next_id}",
                    group = group,
                    bbox = bbox,
                    last_seen_s = timestamp,
                    candidates = deque(maxlen = self.config.confirmation_frames),
                )
                self._next_id += 1
                self._tracks[track.track_id] = track

            else:
                # Update the matched target's position and time without losing its readings.
                track = self._tracks[match]
                track.bbox = bbox
                track.last_seen_s = timestamp

            current_tracks.append(track)

        return current_tracks

    def confirm(self, track, value):
        """Decide whether this target's latest pressure agrees with its recent readings."""

        if value is None:
            # An unreadable pressure breaks the sequence of agreeing measurements.
            track.candidates.clear()
            return "unavailable"

        # Add this pressure to the bounded history; the oldest reading falls out
        # automatically when the configured number of readings is exceeded.
        track.candidates.append(value)

        # Confirm only a full group of readings within the allowed pressure spread.
        # The tiny allowance avoids rejecting a boundary value through rounding error.
        if (
            len(track.candidates) == self.config.confirmation_frames
            and max(track.candidates) - min(track.candidates)
            <= self.config.tolerance_bar + 1e-9
        ):
            # The caller shows the current reading; these values are never averaged.
            return "confirmed"

        return "pending"
