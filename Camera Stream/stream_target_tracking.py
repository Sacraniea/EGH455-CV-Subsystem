"""Associate nearby targets and confirm pressure from consecutive current readings."""

from collections import Counter, deque
from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True)
class TrackingConfig:
    minimum_iou: float = 0.3
    maximum_gap_s: float = 2.0
    confirmation_frames: int = 3
    tolerance_bar: float = 0.2


@dataclass
class TargetTrack:
    track_id: str
    group: str
    bbox: tuple
    last_seen_s: float
    candidates: deque = field(default_factory=deque)


def intersection_over_union(first, second):
    # Compare boxes by their shared area, independently of detection order.
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    overlap = width * height
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - overlap
    return overlap / union if union > 0 else 0.0


class TargetTracker:
    def __init__(self, config=TrackingConfig()):
        self.config = config
        self._next_id = 1
        self.reset()

    def reset(self):
        # Keep the ID counter so a restarted history cannot reuse an earlier ID.
        self._tracks = {}
        self._previous_identity = None

    def associate(self, predictions, identity):
        timestamp = identity.device_timestamp_s
        previous = self._previous_identity

        if not isfinite(timestamp):
            self.reset()
            raise ValueError("Target tracking requires a finite capture timestamp")

        if previous is not None and (
            identity.sequence_number <= previous.sequence_number
            or timestamp <= previous.device_timestamp_s
            or identity.image_size_px != previous.image_size_px
        ):
            # A restarted source or changed image size starts a fresh scene.
            self.reset()

        self._previous_identity = identity
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if timestamp - track.last_seen_s <= self.config.maximum_gap_s
        }
        groups = [
            "valve" if item["class_name"] in ("openValve", "closedValve")
            else item["class_name"]
            for item in predictions
        ]
        choices = [
            [
                track_id
                for track_id, track in self._tracks.items()
                if track.group == group
                and intersection_over_union(track.bbox, item["bbox_xyxy_px"])
                >= self.config.minimum_iou
            ]
            for item, group in zip(predictions, groups)
        ]
        uses = Counter(track_id for options in choices for track_id in options)
        matches = [
            options[0] if len(options) == 1 and uses[options[0]] == 1 else None
            for options in choices
        ]

        # Ambiguous overlap must not lend a confirmed history to another gauge.
        # Remove those old tracks so they cannot keep blocking later clear frames.
        ambiguous = {
            track_id
            for options, match in zip(choices, matches)
            if match is None
            for track_id in options
        }
        for track_id in ambiguous:
            self._tracks.pop(track_id, None)

        for track_id, track in self._tracks.items():
            if track_id not in matches:
                # Even a short absence breaks the run of consecutive readings.
                track.candidates.clear()

        current = []

        for item, group, match in zip(predictions, groups, matches):
            bbox = tuple(item["bbox_xyxy_px"])

            if match is None:
                track = TargetTrack(
                    track_id=f"target-{self._next_id}",
                    group=group,
                    bbox=bbox,
                    last_seen_s=timestamp,
                    candidates=deque(maxlen=self.config.confirmation_frames),
                )
                self._next_id += 1
                self._tracks[track.track_id] = track

            else:
                track = self._tracks[match]
                track.bbox = bbox
                track.last_seen_s = timestamp

            current.append(track)

        return current

    def confirm(self, track, value):
        if value is None:
            track.candidates.clear()
            return "unavailable"

        # Only the current pressure is published; the window is never averaged.
        track.candidates.append(value)
        if (
            len(track.candidates) == self.config.confirmation_frames
            and max(track.candidates) - min(track.candidates)
            <= self.config.tolerance_bar + 1e-9
        ):
            return "confirmed"

        return "pending"
