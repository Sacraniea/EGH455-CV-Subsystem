# GCS integration: logs, detection images and video

This adapter follows the GCS MQTT Sender Tools README provided by Lachlan on
8 October 2026. The current GCS broker on the Pi is `127.0.0.1:1883`.
The publish commands below assume the forwarder runs on that Pi. From the Mac,
use the Pi's current reachable LAN address, `10.88.39.170`, for `--broker`.
Supply the bare IP or hostname, not a `mqtt://` URL or Docker's internal
service name `mqtt`.

| Output | Route | Payload |
| --- | --- | --- |
| Live annotated video | Existing HTTP `/stream.mjpeg` link, fed by local NetGear | JPEG video |
| Sampled ArUco, gauge and valve observations | MQTT `gcs/drone_log` | Exactly `action` and `content` strings |
| Selected annotated image per ArUco sighting or periodically for targets; exact selected file for manual sends | MQTT `gcs/detection` | Exactly `type`, `description`, `mime`, `image`, `timestamp` |
| Environmental telemetry | MQTT `gcs/telemetry` | Owned by the sensor subsystem; not sent by these tools |

The video URL does not deliver structured detection results. NetGear remains an
image transport after composition, as described in the CV integration report.
The GCS compatibility mapping is a separate MQTT output adapter; it does not
change the camera's native result schema or make MQTT an internal frame joiner.

## Run the stream and forwarder

Keep the target-enabled camera demo running in terminal 1. From `Camera Stream`,
using the target environment described in the [stream README](../README.md).
This camera command shows the Mac's HTTP address:

```sh
.venv-targets/bin/python -B run_local_demo.py \
  --camera --targets --profile 720p30 \
  --http-host 10.88.38.220 --max-frames 18000 --open-browser
```

On the Pi, use `--http-host 10.88.39.170` and view
`http://10.88.39.170:8080/stream.mjpeg`. LAN addresses can change.
Choose the frame limit you need; it is a processed-frame count, not a duration.
The demo also enables ArUco and accepts dictionary IDs 0–99 by default. Omit
`--permitted-ids` to keep all IDs accepted; an explicit list restricts pose
acceptance to those IDs. Metric marker pose requires the actual measured
marker size and matching camera calibration. Without calibration, the marker log
still shows its ID, image centre and a reason that pose is unavailable. Target
pressure uses the gauge reader's separate dial calibration.

In terminal 2, from the same folder, first preview:

```sh
.venv-targets/bin/python -B run_gcs_forwarder.py --timeout-s 30
```

This subscribes to local `127.0.0.1:18883`, topic `taip/local/results`, and saves
prepared payloads in a new `diagnostics/gcs/*.jsonl` file. It does not open any
connection to the GCS. The terminal shows each prepared `action` and `content`,
the preview/publish mode and the evidence-file path. The JSONL keeps each payload
with its source identity and delivery outcome. The forwarder can also run in the
existing `.venv`; only the camera sender needs the target model dependencies.

To send to the agreed GCS broker:

```sh
.venv-targets/bin/python -B run_gcs_forwarder.py \
  --publish --broker 127.0.0.1 --port 1883 \
  --interval-s 5 --timeout-s 600
```

If an older forwarder is already running, stop only that process with Ctrl+C
and start the updated command. The target stream, local broker and browser video
can stay running. Use one log forwarder to avoid sending duplicate samples. The
publish startup line identifies the selected broker and `gcs/drone_log`.

The source port and topic can be changed with `--local-port` and `--local-topic`.
The GCS log destination topic stays fixed to the agreed contract. The forwarder
does not start or stop the camera, local broker, video receiver or GCS. Ctrl+C
stops only the forwarder. If the local demo ends, its broker disconnect will
end the forwarder with a reported error; that does not undo already acknowledged
messages. Start it again alongside the next demo session.

The current GCS connection uses MQTT 3.1.1, QoS 1, no retained messages, and no
credentials or TLS, matching the supplied README's connection example. If the
GCS requires credentials or TLS, add that configuration before publishing.
Remote socket/PUBACK failures are bounded and stop the forwarder. Hostname DNS
resolution is controlled by the operating system. No automatic stale replay
or reconnect is attempted; the camera continues independently. Incoming results
use the existing bounded subscriber queue; queue overflow reports an error.

## What the log contains

The outer payload is still exactly two strings, `action` and `content`.
`content` now contains plain text for the GCS log view, rather than a JSON
record embedded in a string.

| Action | Meaning |
| --- | --- |
| `ArUco detected` | At least one marker has an accepted pose. Other markers in the same message can still have unavailable poses. |
| `ArUco detected - pose unavailable` | Markers were decoded, but every pose was withheld. |
| `ArUco processing error` | The ArUco processor failed; this is distinct from a normal pose rejection. |
| `Gauge detected` | One detected gauge, with its accepted pressure or `Pressure unavailable`. |
| `Valve detected` | One detected valve, labelled open or closed from the model's class. |
| `Target processing error` | The target processor failed; this is distinct from an unreadable gauge. |

For example, an accepted pose could render as follows. These values illustrate
the format; they are not hardware measurements.

**Action:** `ArUco detected`

**Content:**

```text
Marker 7: 10.0 cm right, 20.0 cm down, 200.0 cm forward; image position (40, 40) px.
Frame 42 | Device time 12.500 s | Run run-abc
```

A rejected pose instead uses `ArUco detected - pose unavailable`, with text such
as `Marker 7: pose unavailable (corner fit error too high: 4.00 px)`, followed by
the image position and identity footer. Synthetic summaries begin with
`Synthetic demo.`.

Each marker summary identifies its ID and gives:

- An accepted camera-relative position in **centimetres**, using right/left,
  down/up and forward directions, or a friendly reason why pose is unavailable.
- Its image centre in pixels, calculated as the arithmetic mean of its detected
  corner coordinates in the source image.

The position directions follow `camera_optical`: x is right, y is down and z is
forward. A forward distance is z depth, not Euclidean range. Centimetres are
a display conversion; native pose translations remain in metres. A footer records
the frame sequence, device monotonic timestamp in seconds and the full run ID.
Those monotonic seconds are not a UTC time of day. Synthetic data is explicitly
marked so a demonstration cannot be mistaken for a camera measurement.

Gauge and valve messages are deliberately short. For example:

```jsonl
{"action": "Gauge detected", "content": "Pressure: 2.9 bar; image position (40, 30) px; Frame 42."}
{"action": "Valve detected", "content": "Open valve; image position (40, 30) px; Frame 42."}
```

A gauge reports pressure only for an accepted reading, otherwise
`Pressure unavailable`. Pressure uses up to two decimal places, without trailing
zeros. A valve reports `Open valve` or `Closed valve`. Each includes the box
centre in source pixels and the frame number; an unusable box reports
`image position unavailable`. A failed target branch reports
`Target processing failed; Frame 42.` under `Target processing error`.
Synthetic target text starts with `Synthetic demo.`.

These target messages omit the ArUco-style device-time/run footer. Full source
timestamps remain in the native frame record, and the audit retains the run ID
and frame number. Unrecognized model classes do not produce a target GCS log.

Each selected frame can produce the unchanged combined ArUco message plus one
separate message for every gauge or valve. Two gauges therefore keep separate
pressure results. Empty frames generate no detection log. A decoded marker with
a rejected pose and a gauge with an unavailable pressure remain visible in their
logs; neither is replaced with a made-up measurement.

Full native frame evidence remains in the local sender's `frames.jsonl`
(`sender/frames.jsonl` inside a demo run). It retains the source timestamps,
image geometry, complete corners, accepted/rejected pose fields, all target boxes
and gauge readings. `run.json` keeps camera/model/gauge settings. Captured image
sidecars also retain the full record. Automatic GCS image descriptions keep the
existing ArUco summary, as described below.

**Receiver compatibility:** the outer MQTT JSON object is unchanged, but log
`content` is no longer JSON. Any consumer that previously called `JSON.parse`
or `json.loads` on that string must adapt. Display it as text. Structured pose
columns, filtering or pose widgets require a jointly agreed data schema; this
human-readable summary is not a replacement machine-readable pose interface.
The GCS also needs to accept the action strings listed above.

The forwarder selects at most one frame every five seconds by default, then
sends all of that frame's prepared messages. The interval applies to frame
selection, so several small logs can arrive together. Changing poses or targets
between selected frames can be omitted; this is a sampled commissioning log,
not a guarantee that every encounter is logged. The full per-frame record stays
in the camera session. Set a different positive `--interval-s` only to match the
agreed logging rate. Camera frame rate is independent.

Duplicate/out-of-order source frames are suppressed using the highest sequence
for the 16 most recently seen runs. QoS 1 can still duplicate a destination
delivery. ArUco text includes frame and run identity; short target text includes
the frame number, with the run ID retained in the local audit. There are no
dedicated structured deduplication fields in this two-string schema. Agree on a
machine-readable identity contract if GCS-side deduplication is required. No global exactly-once database guarantee is claimed across bridge
restarts.

Every prepared message has a separate local audit line marked `preview`,
`broker_acknowledged`, `failed` or `not_attempted`. If one publication fails, the
forwarder stops sending, retains earlier acknowledgements and marks the rest of
that selected frame's messages `not_attempted`. It then stops without retrying.
The audit's `prepared_at_utc` is preparation time, not camera capture time. A
broker acknowledgement does not prove the GCS subscriber parsed the payload or
saved it to MySQL.

## Send an original image file explicitly

The camera's structured MQTT record contains no image. The GCS image helper
therefore takes an existing original PNG/JPEG file and sends it only when
requested. It does not automatically capture images from the live feed.

Preview and optionally save the complete payload:

```sh
.venv/bin/python -B send_gcs_detection.py /path/to/original.png \
  --type Valve --description "Original full-frame valve capture" \
  --output /path/to/new-payload.json
```

Publish the selected file:

```sh
.venv/bin/python -B send_gcs_detection.py /path/to/original.png \
  --type Valve --description "Original full-frame valve capture" \
  --publish --broker 127.0.0.1 --port 1883
```

The helper preserves original file bytes, image resolution and embedded file
metadata. It validates PNG/JPEG bytes, then Base64-encodes those same bytes.
It does not resize, crop, rotate, annotate or re-encode the image. Preview prints
the byte length and SHA-256 without dumping Base64 into the terminal. Existing
output files are never overwritten.

The optional `--timestamp` accepts a known timezone-aware ISO 8601 event time.
If omitted, the field is current UTC payload-preparation/publication time;
it is not inferred from the file or a monotonic camera timestamp. Confirm this
meaning and allowed `type` values with the GCS receiver.

## Automatic images: ArUco sightings and periodic targets

The camera selects one frame every three seconds while gauges or valves are
visible. Each selection saves the untouched full-resolution `frame-N.png` and
a full-resolution `frame-N.annotated.png` from the same frame's composed output.
The annotations include current targets, including gauges whose pressure is
unavailable, without another detector pass. ArUco keeps the existing
once-per-sighting policy. When both policies select the same frame, they share
one event and one image upload. A second process uploads the annotated copy,
independently of the annotated MJPEG preview. GCS retains that copy; originals
stay on the Pi.

Add `--capture-detections --capture-interval-s 3` to the existing camera demo
command, retaining the same marker size, calibration and HTTP host options.
Enable `--targets` for gauge and valve detection. The launcher prints the
capture directory, underneath `sender/detections` in that run's evidence folder.
For a repeatable example, choose a new output directory each run:

```sh
.venv-targets/bin/python -B run_local_demo.py \
  --camera --targets --profile 720p30 --http-host 10.88.38.220 \
  --capture-detections --capture-interval-s 3 --max-frames 18000 \
  --marker-side-length-m 0.20 \
  --calibration "../ArUco Detection/calibration/oak_cam_a.json" \
  --calibration-mode oak-1080 \
  --output-dir diagnostics/local-network/target-photo-demo-001
```

This example assumes a measured 200 mm marker. If the displayed marker measures
150 mm across its outer black square, use `--marker-side-length-m 0.15`.
All dictionary IDs 0–99 remain permitted because no explicit list is supplied.
The camera example uses the Mac's HTTP address; use `--http-host 10.88.39.170`
on the Pi, or update either address if it has changed. Wait for the model and
camera to start before starting the uploader against the printed directory.

Preview pending images without contacting the GCS:

```sh
.venv-targets/bin/python -B run_gcs_image_forwarder.py \
  --capture-dir diagnostics/local-network/target-photo-demo-001/sender/detections \
  --timeout-s 30
```

To upload the captures, explicitly enable publication:

```sh
.venv-targets/bin/python -B run_gcs_image_forwarder.py \
  --capture-dir diagnostics/local-network/target-photo-demo-001/sender/detections \
  --publish --broker 127.0.0.1 --port 1883 \
  --interval-s 3 --max-mbps 1 --publish-timeout-s 60 --timeout-s 0
```

`--publish-timeout-s` controls the image publisher's acknowledgement wait and
defaults to 60 seconds. Full-resolution images can occupy several megabytes;
their transmission and broker acknowledgement need time. This setting does
not guarantee delivery within 60 seconds, and it leaves the stream and log
publishers' two-second default unchanged.

`--timeout-s 0` runs until Ctrl+C. A positive value sets a runtime deadline
checked between uploads, with 600 seconds retained as the default. Startup
prints the broker, topic, resolved capture directory, runtime and acknowledgement
wait. Waiting or pacing status appears at 30-second intervals between uploads
with pending counts, and shutdown prints a summary. Check that the shown folder
matches the current camera run: the uploader watches only the directory passed
to `--capture-dir` and does not follow new run folders
automatically. Restart it with the new run's printed path when the camera
session changes.

The log forwarder is separate and runs alongside this image uploader.
Automatic GCS detection `description` uses the same readable summaries as the
logs. It can describe ArUco, gauges, valves, or several together. For example,
an illustrative ArUco caption could be:

```text
Marker 7: 10.0 cm right, 20.0 cm down, 200.0 cm forward; image position (40, 40) px.
Frame 42 | Device time 12.500 s | Run run-abc
```

ArUco captions include accepted camera-relative position in centimetres or a
friendly pose-rejection reason, the source-pixel centre, and the existing
frame/time/run footer. Gauge captions say `Pressure: 2.9 bar`,
`Pressure unavailable`, or `Pressure unavailable (awaiting confirmation)`.
Valve captions say `Open valve` or `Closed valve`. Target captions include
image position and frame number. Unconfirmed raw pressure is never included.

The `type` is `Gauge`, `Valve`, `ArUco`, or a comma-separated combination
such as `Gauge, Valve`. This keeps one photo for a frame containing multiple
targets. Confirm the GCS accepts these type strings; its shared contract defines
the field as text, but local tests do not verify its dashboard/database rules.
The uploader verifies each selected observation against the saved frame record
before preparing a caption.

The outer image message still has exactly `type`, `description`, `mime`, `image`
and `timestamp`. New automatic events send the annotated PNG's exact file
bytes, keeping the MIME type and timestamp meaning unchanged. The `timestamp`
is the host's capture-selection time in UTC, not a fabricated conversion of
the device monotonic timestamp to UTC exposure time. The complete same-frame
record and `event_id` remain in the local `*.event.json` sidecar.
`target_results` records selected gauge/valve evidence; `marker_ids` can be
empty for a target-only photo. Delivery receipts retain the event identity and
the uploaded image filename/hash: the annotation for new captures, or the
original for older captures without an annotation descriptor.

The sidecar remains `schema_version: 1`. Its top-level `image_file`,
`image_sha256` and `image_bytes` describe the original PNG. The optional
`annotated_image` object contains those same three fields for the rendered PNG.
The uploader selects that descriptor when present and fails the event if it or
its file is corrupt; it does not silently upload the original instead. Older
captures without `annotated_image`, including marker-only sidecars, continue
to upload their original bytes. Manual `send_gcs_detection.py` behavior is
unchanged: it sends the exact selected file.

The GCS must display `description` as plain text. Consumers that parsed it as
JSON must adapt; it no longer contains structured pose fields or an `event_id`
key. Existing GCS history is not retroactively annotated, rewritten or replayed.
Restart the camera and both forwarders after this update so each loads the new
behavior.

The capture policy is:

- ArUco: confirm visibility for 0.3 seconds, then capture once per sighting,
  even if pose is withheld. Require a two-second absence and a ten-second
  cooldown before another sighting of the same marker can trigger a photo.
- Gauge/valve: wait for 0.3 seconds of target visibility, then select at most
  one frame every three seconds. `--capture-interval-s` adjusts the interval.
  All current targets share the selected frame. A marker-triggered photo also
  includes current targets and resets their three-second timer.
- Unreadable or unconfirmed pressure does not remove the gauge from a photo.
- Queue-full attempts are paced too; they do not retry on every camera frame.
- Retain at most three pending image events and at most 256 MiB of PNG bytes
  per session by default. Both original and annotated files count toward the
  storage limit; each pair uses one pending slot. `--capture-max-pending` and
  `--capture-storage-mib` configure these bounds.

The dictionary/ID is the ArUco identity. Repeated physical markers with the same
ID cannot be distinguished reliably by that key. Periodic target photos do not
require a unique asset ID. The stream's local box matching is used to confirm
gauge readings; it does not establish which physical asset is in view.

PNG encoding is lossless. The source pixels remain untouched, and the annotated
copy uses the existing full-resolution composition from that same frame.
Capture encoding/disk work for both files is synchronous and can briefly affect
capture throughput; network waits happen in the separate uploader. Metadata is
committed last as an atomic `*.event.json` readiness signal. The image is never
paired with an independently fetched latest result.

Queue/storage limits deliberately trade image coverage for bounded resource
use. A capture that cannot be retained is recorded as skipped, and full frame
measurement logs remain available. Originals are never automatically deleted.
Inspect the capture audit/summary for skipped or failed events; do not claim all
target sightings have images if those counts are nonzero.

## Bandwidth and gallery thumbnails

NetGear currently transfers on loopback. Over the LAN, the HTTP MJPEG video and
MQTT uploads share the same network interface. Separate processes avoid making
the camera wait for MQTT, but do not reserve network bandwidth for video.

The uploader sends one image at a time. Its default pacing remains ten seconds;
the command above explicitly selects the supported minimum of three seconds.
The previous complete JSON/Base64 payload size extends that interval to budget
roughly 1 Mbit/s of average payload traffic. A three-second capture interval does
not guarantee delivery every three seconds. For example, a 2.43 MB PNG
becomes about 3.24 MB of Base64, requiring roughly 26 seconds at that budget.
When the pending queue is full, further photo attempts are recorded as skipped;
processed-frame detection results are still retained. A single
MQTT message can still transmit in a burst; this is not traffic shaping or a
guarantee of uninterrupted video on every Wi-Fi network. Network framing,
retransmissions and GCS-to-browser traffic are additional costs. Measure video
latency and received FPS during commissioning and lower the upload budget if
needed.

The default maximum payload is 12 MB. Invalid or oversized events are recorded
as failed and are not retried automatically. Sent/failed receipts release
pending queue slots while preserving both PNG files. A preview never
consumes a pending event. Confirm the broker's packet-size limit against the
configured payload limit; a longer acknowledgement wait cannot overcome that
limit.

If connection setup fails before publication, or the publisher confirms that
the image was never handed to the MQTT client, the same event remains pending
and is retried after the configured pacing delay. The latest diagnostic is
saved in `*.event.retry.json`; it does not release the event's queue slot and is
removed when a terminal sent/failed receipt is written. Once the image has been
handed to MQTT, an error or missing acknowledgement leaves delivery uncertain.
That event receives a failed receipt and is not automatically replayed, because
the GCS may already have received it.

Run one image uploader per capture directory. Receipts prevent routine resends
after restart; MQTT QoS 1 can still duplicate delivery. The sidecar's full run ID
and frame sequence allow correlation with local evidence, but the five-field
image contract has no dedicated machine-readable event identity field. GCS-side
deduplication using such a field needs a jointly agreed schema. Capture selection
is not an exactly-once database guarantee. A sent receipt confirms
broker acknowledgement, not database ingestion. A later event can open a fresh
connection after an earlier upload failed with uncertain delivery; the earlier
failed event remains excluded from automatic replay.

Base64 uses four encoded characters for three original bytes, approximately
33% overhead ([RFC 4648](https://www.rfc-editor.org/rfc/rfc4648.html#section-4)).
For example, a 1 MB image becomes approximately 1.33 MB before the surrounding
JSON. One such image every ten seconds averages about 1.07 Mbit/s before other
network overhead; the size-based pacing extends that interval for a 1 Mbit/s
budget.

The GCS retains the uploaded full-resolution annotated copy for new automatic
events and uses CSS to display it smaller. CSS sizing does not reduce the bytes
downloaded by the browser. For a large gallery, the GCS can store an additional
derived thumbnail and load the full-resolution annotated image only on demand.
That would be a GCS-side display/API change. Untouched originals remain on the
Pi; automatic annotation does not resize the image. See [MDN's responsive-image guidance](https://developer.mozilla.org/en-US/docs/Web/HTML/Guides/Responsive_images).

## Verify with the GCS

The local automated tests check exact outer schemas, real MQTT packet exchange,
sampling, per-target log formatting, rejected-reading preservation and
byte-for-byte image payloads. They do not contact Lachlan's broker or establish
GCS/MySQL delivery.

For commissioning, run the camera and preview first, then explicitly publish.
Lachlan should confirm an event in his log UI/database and inspect its content
against the local audit's run ID and sequence. For a selected image, confirm its
resolution and compare the decoded bytes/hash with `frame-N.annotated.png`
for a new automatic capture, or the selected original file for an older capture
without that descriptor or a manual send. Broker reachability, GCS action/type
acceptance, field-size limits and actual database ingestion are only verified
by that joint test.
