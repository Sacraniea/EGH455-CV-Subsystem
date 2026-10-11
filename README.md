# EGH455 Computer Vision Subsystem

This subsystem takes frames from the OAK-D Lite camera, detects gauges, valves
and ArUco markers, and sends the video, readings and detection photos to the GCS.

## Structure

| Folder | What it does |
| --- | --- |
| `Camera Stream` | Runs the camera, brings the detection results together, draws the live feed and sends results to the GCS. |
| `Target Detection` | Finds gauges and valves, supplies the gauge crops and draws the detection boxes. |
| `Gauge Reader` | Takes each gauge crop, finds the needle and pivot, and converts its direction into pressure. |
| `ArUco Detection` | Reads marker IDs and estimates their position using the camera calibration and marker size. |

The camera supplies one frame to both detection paths. Gauge crops go straight
from the target detector to `read_gauge()`. The stream keeps the results with their
frame and draws annotations on a copy, leaving the original image unchanged.
If a gauge cannot be read, we keep the detection and report pressure unavailable.

Start with [run_local_demo.py](Camera%20Stream/run_local_demo.py) to see how the
processes are started. [stream_target_detection.py](Camera%20Stream/stream_target_detection.py)
connects target detection to the gauge reader.

## Setup

Install [Pixi](https://pixi.prefix.dev/latest/installation/), then open a terminal:

```sh
git clone https://github.com/Sacraniea/EGH455-CV-Subsystem.git
cd EGH455-CV-Subsystem
pixi install --locked
```

If you already have the repository, open that folder and run `pixi install --locked`.
Pixi installs Python and the required libraries from the saved environment.
Run the commands below from this folder, where `pixi.toml` is located.

Keep the four subsystem folders together. When moving to the Pi or another
computer, install with Pixi there rather than copying the `.pixi` folder.

## Run

Connect the camera, then start the full system locally:

```sh
pixi run cv-subsystem --open-browser
```

This starts the camera, detections, live video and both GCS forwarders. Without a
GCS address, logs and photos are prepared locally but nothing is sent to the GCS.
The video opens at `http://127.0.0.1:8080/stream.mjpeg`.

To send logs and annotated detection photos to the GCS:

```sh
pixi run cv-subsystem GCS_IP_ADDRESS --open-browser
```

Replace `GCS_IP_ADDRESS` with the address of the computer hosting the GCS broker.
Use `127.0.0.1` if the subsystem and GCS broker are running on the same Pi.
MQTT uses port **1883**. Both forwarders start automatically and use the current
run's results and photos. **Ctrl+C stops the whole run.**

To make the video available to the GCS or another device on the same network:

```sh
pixi run cv-subsystem GCS_IP_ADDRESS --http-host CAMERA_COMPUTER_IP
```

Replace `CAMERA_COMPUTER_IP` with the address of the computer running the camera.
Set the GCS video URL to `http://CAMERA_COMPUTER_IP:8080/stream.mjpeg`.
Video uses this link; MQTT carries the logs and detection photos.

The main command uses 720p at a requested 30 FPS, the saved OAK calibration and
**200 mm ArUco markers**. Detection processing can reduce the frame rate.
For another marker size, add `--marker-side-length-m` with its measured black
square size in metres. The gauge reader is set for an upright **0–10 bar** dial.
A needle up to 10° outside either scale end reads as 0 or 10 bar; larger offsets
remain unavailable. This tolerance does not correct camera rotation.

## Quick checks

| Command | What it checks |
| --- | --- |
| `pixi run cv-sys-synthetic-test` | Video and messages using generated frames; no camera needed. |
| `pixi run cv-sys-camera-test` | Camera capture over 90 frames. |
| `pixi run cv-sys-target-detection-test` | Live gauge and valve detection; press q to stop, or it finishes after 3000 frames. |
| `pixi run cv-sys-local-test` | Full local camera run over 900 frames, without sending to the GCS. |

## Saved results

Each run prints its folder under `Camera Stream/diagnostics/local-network/`.
Inside it, `sender/frames.jsonl` holds the detection results, `sender/detections/`
holds original and annotated photos, and `logs/` holds the process output.
`gcs-logs.jsonl` records the log forwarding results. Files stay there after stopping;
Ctrl+C does not wait for queued photos to finish uploading.

For more detail, see the [stream walkthrough](Camera%20Stream/docs/stream-walkthrough.md)
or the [GCS integration guide](Camera%20Stream/docs/gcs-integration.md).
