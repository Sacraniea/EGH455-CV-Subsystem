# ArUco detection and camera-relative pose

For the shared stream and Aidan's model adapter, start with the
[computer vision handover](../README.md). This README covers the ArUco component.

The standalone ArUco software supports generated images, image files and the existing OAK colour stream. It decodes `DICT_5X5_100`, preserves frame evidence, estimates marker-in-camera pose, withholds rejected poses, displays a copy of the frame, and saves strict JSON records. It does not run Aidan's detector, extract gauge values, publish MQTT, or calculate UAV body/world coordinates.

Software completion is separate from physical validation. Live ID 7 detection and approximate screen-marker distance checks have been demonstrated. Controlled 1–3 m accuracy, motion/occlusion, Raspberry Pi throughput and endurance remain pending.

## Environment and tests

Run these commands from this `ArUco Detection` directory. The sibling `Camera Stream/.venv` already provides Python, NumPy, OpenCV and DepthAI. No new dependency is required in that environment. `requirements.txt` records the existing pinned versions for a fresh environment; install only one OpenCV distribution, with GUI support for the live preview.

```bash
PYTHONPATH="../Camera Stream" "../Camera Stream/.venv/bin/python" -B -m unittest discover -s tests
```

The live executable finds sibling stream modules using its own location, so it does not require `PYTHONPATH`. The test suite uses the original stream `FramePacket` directly and still needs the path above. The shared camera runner imports these same modules; `.venv-targets` can also run the ArUco tests and commands.

## Live preview: current screen experiment

Close any other process using the OAK. This example uses a screen marker physically measured at 150 mm across its outer black square; browser zoom changes that size. Any ID from 0 to 99 in `DICT_5X5_100` is accepted.

```bash
"../Camera Stream/.venv/bin/python" -B run_aruco_live.py \
  --marker-side-length-m 0.15 \
  --profile 1080p30 \
  --calibration calibration/oak_cam_a.json \
  --calibration-mode oak-1080
```

Press **q** in the preview to stop, or Ctrl+C in the terminal. Use `--max-frames 300` for a bounded trial; this is 300 **processed** frames, not a promise of ten seconds. The actual elapsed time and sequence gaps are recorded. The live view shows ID, acceptance/rejection, X/Y/Z, straight-line range and reprojection RMS. It never draws into the source frame.

For the assignment's measured **200 mm printed marker**, change `--marker-side-length-m` to `0.2`. This is the default, but explicitly recording the intended size in the command is useful. Omitting `--permitted-ids` accepts all 100 dictionary IDs. Add `--permitted-ids 7 23` only to restrict accepted poses to those two IDs; the dictionary is fixed to the FAQ's `DICT_5X5_100`.

Without `--calibration`, detections remain available with `MISSING_CALIBRATION` and no pose. Omit `--calibration-mode` too in that case.

The RMS limit defaults to **1.0 px** and can be set with `--max-reprojection-error-px`. It is provisional, not an empirically validated accuracy guarantee. Do not raise it solely to make a rejected observation pass.

## Calibration geometry

`exact` is the safe default: use the supplied matrix unchanged. A frame-size mismatch yields `CALIBRATION_SIZE_MISMATCH` for decoded permitted markers, preserving their pixel evidence.

`oak-1080` is an explicit transform for the **existing** `Camera Stream/depthai_source.py` pipeline (`THE_1080_P`, default ISP, `.video` output):

- Native calibration at 3840×2160 → scale intrinsics to 1920×1080.
- Native calibration already at 1920×1080 → retain it.
- `1080p30` → no further transform.
- `720p30` → centre crop the 1920×1080 geometry at `(320, 180)` to 1280×720; do not scale focal lengths again.

Other native sizes are rejected in this mode. This encodes the selected camera-mode assumption; matching dimensions and a low residual do not independently verify the field of view or physical accuracy. Revisit it if sensor mode, ISP scaling, video output, rectification, focus, lens, or camera changes. Supply calibration for the actual geometry using `exact` when appropriate.

Live startup verifies the calibration's CAM_A perspective model and camera MXID against the connected device. The run manifest embeds the original calibration JSON, SHA-256, transform name and effective calibration. All 14 factory distortion coefficients are retained. Image processing cannot independently verify which camera took a photograph.

Both live runners use `load_stream_calibration()` in `aruco_calibration.py` to
load the selected geometry and prepare this evidence. `load_calibration()` and
`prepare_stream_calibration()` remain available for callers needing those steps
separately.

## Saved evidence

Each live invocation creates `runs/<UTC time>-<unique suffix>/`, or a **new** directory supplied with `--output-dir`.

- `frames.jsonl`: one JSON object per processed frame, flushed after writing. Empty detections still retain sequence, timestamps, image dimensions and processing time.
- `run.json`: software versions, camera identity, requested profile, calibration evidence, marker settings, counts, failure reasons, timing summary and stop reason. Status starts as `running`, then becomes `complete`, `interrupted` or `failed`. Manifest updates are atomic. A hard kill may leave `running` and a partial last JSONL line; do not interpret it as a completed trial.

Files are not overwritten on a subsequent run. A disk/write failure stops processing and closes resources. If the disk cannot accept the final manifest, the command returns failure; the last manifest may remain `running`. Flushing is not a guarantee against power loss.

No raw video is retained by this runner. For visual evidence, capture a screenshot or run a saved frame through the image command. JSONL records contain source corners for audit.

`processing_time_ms` measures detection and pose only. It excludes overlay, serialization, disk writes, acquisition and display. `stream_summary.received_fps` is observed application consumption throughput with ArUco enabled, not independent sensor FPS. The inherited latency metric is DepthAI frame timestamp to application dequeue, not glass-to-glass latency. Source queue drops appear as sequence gaps. Frame metadata uses the stream's existing DepthAI clock domain, not UTC; UTC run start/end are separate fields.

The recorder streams evidence without accumulating images or results. The shared stream controller now uses bounded online diagnostics: whole-run counts and mean latency, with latency P95 over the most recent up to 2048 processed frames (the summary records the window and sample count). Tests do not establish a Raspberry Pi performance rate.

## Process one image

```bash
"../Camera Stream/.venv/bin/python" -B aruco_image.py photo.png \
  --output-dir runs/photo-check \
  --marker-side-length-m 0.2
```

Add `--calibration path/to/calibration-for-this-image.json` for pose estimation. Image calibration is always used as supplied; it is not automatically resized or cropped. Output is `overlay.png` plus `results.json`. Capture timestamps and sequence number are `null` because a standalone image lacks that evidence. Marker rejection is a normal completed command (exit 0 with an explicit failure reason), while I/O/configuration errors return nonzero. `results.json` is published last; its presence indicates completed image output. An output-write failure may leave a partial output directory for inspection.

## Evaluate a recorded run

```bash
"../Camera Stream/.venv/bin/python" -B evaluate_aruco_run.py runs/YOUR-RUN/frames.jsonl
```

This reports empty-frame counts, per-ID accepted/rejected detections, range mean/population standard deviation/min/max/P95, reprojection errors and processing-time statistics. It does not label consistency as physical accuracy.

For a **stationary run at one measured camera-to-marker-centre distance**:

```bash
"../Camera Stream/.venv/bin/python" -B evaluate_aruco_run.py runs/YOUR-RUN/frames.jsonl \
  --reference-distance-m 0.30 --output runs/YOUR-RUN/evaluation.json
```

The reference applies to the entire file. Do not supply one reference for a moving trial. Rejected poses are excluded from distance statistics but counted explicitly; accepted-only accuracy therefore does not describe detection availability. Malformed/truncated records produce a line-numbered error rather than silently disappearing from statistics.

## Integration contract

For computation alone:

```python
from aruco_processing import process_aruco_frame

observations, frame_record = process_aruco_frame(
    packet, calibration, config, max_reprojection_error_px=1.0
)
```

`packet` follows `FramePacketLike`: BGR `uint8` image plus sequence number, device timestamp and host-receipt timestamp. It can be the same original `FramePacket` passed to Aidan's detector and the gauge branch. The computation neither opens a camera/window nor writes files. `read_aruco()` remains available for callers wanting only typed marker results.

The frame record is schema version 1, producer `aruco`, corners in `source_px` `(x,y)` coordinates, source dimensions, original identity, processing time, configuration, calibration and `results`. Per-marker records retain the earlier schema. `ArUcoDisplay` connects processing to the existing `run_stream()` display contract and accepts an `on_result(record)` callback. The controller owns source/display cleanup; the runner owns the recorder. Processing/logging failures propagate after cleanup and are recorded as failed runs.

For accepted poses, translation is **marker centre in the camera optical frame**, in metres: +X right, +Y down, +Z forward. Range is `sqrt(X²+Y²+Z²)`. Rotation is a Rodrigues axis-angle vector in radians transforming marker coordinates into camera coordinates, not Euler yaw/pitch/roll. No camera-to-body transform is invented.

`valid` means **pose accepted by the implemented checks**, not “marker decoded with this probability”. An invalid result can still contain a decoded ID and reliable corners. Pose estimation uses IPPE_SQUARE followed by LM refinement and corner RMS reprojection testing. Planar ambiguity, wrong physical size and imperfect calibration can still produce plausible low-residual poses.

Expected rejection reasons include disallowed ID, duplicate ID, missing calibration, calibration-size mismatch, solver failure, non-finite pose, non-positive depth and excessive reprojection error. Repeated IDs in one frame retain all observations in deterministic order but withhold all poses for that ID; there is no instance tracker to disambiguate them. This prevents the screen-preview feedback seen during development from silently generating multiple distances for one ID. A single resized screen copy can still pass: always keep preview copies outside the camera view.

Malformed images and invalid configuration raise clear exceptions at the programmatic boundary; they are not confused with valid empty frames. Strict serialization replaces unavailable non-finite diagnostic RMS with null and never exports a rejected pose.

## Validation remaining

- Measure the outer black square accurately; white margin is excluded. A scale error in marker size produces proportional translation error.
- Record fixed-distance trials at 1 m, 2 m and 3 m for both profiles, with front-on/oblique views, motion, glare and partial occlusion.
- Separate Z depth from straight-line range when comparing a ruler/tape measurement.
- Test on the Raspberry Pi and record throughput/endurance before claiming onboard requirements.
- Validate ArUco alongside Aidan's detector and gauge overlays in the shared stream; the standalone commands keep their ArUco-only role.

The checked-in factory calibration is specific to MXID `1844301031F2EA0F00`. The screen trial used an approximately 150 mm marker and a rough 30 cm ruler reference; longer distances were estimated. These are development demonstrations, not formal accuracy evidence.
