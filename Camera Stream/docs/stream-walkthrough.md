# Walkthrough: follow one frame

Use this alongside the implementation. We can stop at each checkpoint and trace
an actual packet together before changing anything.

## 1. Start with the entry point

Read `run_camera_stream.main`. `--aruco` selects marker processing and `--targets`
selects Aidan's gauge/valve detector plus gauge reading. Either branch can run
alone, or both can process the same camera packet. The [stream README](../README.md)
has the target environment setup and direct-preview/browser-demo commands.

The integration report's sections 6, 11 and 13 explain the original shared-frame
design. The current runner and adapters are the implementation to follow now.

## 2. Who opens the camera?

Read `run_camera_stream.main`, then `DepthAIFrameSource.__init__`.
The command selects a profile and prepares its processors before opening the
camera. With `--targets`, `prepare_targets()` loads the local model once, checks
its classes and records the model/settings metadata. The command then reserves
a session directory and opens one device.
CAM_A video feeds XLinkOut; the host queue has capacity four. Its non-blocking
setting means the device producer can drop queued frames rather than build an
unbounded backlog. The new command polls `tryGet` with a frame-arrival timeout.

**Checkpoint:** Does a 30 FPS setting guarantee that our Python model processes
30 frames each second? Which recorded fields would reveal a slower consumer?

## 3. What travels through the subsystem?

Read `FramePacket`, then `frame_packet_from_message`.
The image travels with its sequence number and two timestamps. The device
synchronised clock and host dequeue clock share a domain; UTC session times are
separate labels and must not be subtracted from those monotonic timestamps.

**Checkpoint:** Why would passing just an image to an asynchronous consumer make
it harder to attach its answer to the correct frame later?

## 4. Where does processing happen?

Read the loop in `run_stream`, then `process_frame`.
The loop retains Frame N. Each configured processor receives exactly that packet.
The CLI registers the requested ArUco and/or target processors. Calls are
sequential; “independent branches” describes their data dependencies, not
simultaneous threads.
The pixel buffer is temporarily read-only. A frozen dataclass alone would not
prevent someone modifying pixels inside its array.

`ArUcoProcessor.process` calls the existing `process_aruco_frame`. We did not
rewrite marker detection or open a second OAK. Its result has two parts: JSON-safe
evidence for logging, and typed observations for the renderer.

`TargetDetectionProcessor.process` calls `predict_detections()` once. Aidan's
predictor supplies all current-frame boxes and independent clean gauge crops;
the adapter passes each gauge crop to `read_gauge()`. Valves keep their class and
box. Rejected gauge readings also stay in the results. An empty prediction list
produces an empty target result, without reusing a previous gauge crop.

`TargetTracker` associates overlapping boxes across consecutive frames. Three
accepted readings must agree within 0.2 bar before the current pressure is shown.
The untouched reader result stays in `raw_gauge_reading`; `gauge_reading` and
`reading_status` explain what passes the current consistency check. A missing, ambiguous
or rejected observation resets agreement. An uncertain frame never borrows an
old pressure.

**Checkpoint:** If ArUco cannot calculate pose, why should the marker ID and
corners still appear in its structured result?

## 5. How do we prevent wrong-frame annotations?

Read `FrameIdentity` and the checks in `process_frame` and `ResultCompositor`.
Identity includes sequence, both timestamps and image dimensions. A processor
returning Frame N−1's result cannot annotate Frame N, even if a sequence number
happened to match after a restart. Saved records also include a unique run ID.
The controller completes the configured result set before composing; each normal
rejection counts as a completed branch.

**Checkpoint:** Which two values would you inspect first if an annotation looked
one frame behind the object?

## 6. Why copy at composition?

Read `ResultCompositor.compose`, then each processor's `draw` method.
The compositor copies the original. Renderers draw only on the overlay and
return it in a new packet retaining the original timestamps and sequence.
ArUco's existing renderer makes an additional temporary copy; it preserves
annotations drawn by earlier renderers and keeps its standalone API unchanged.
The display receives the overlay; the source remains suitable for verification
and every subsequent algorithm.

For targets, `draw()` passes saved labels to Aidan's `annotate_image()` on the
accumulated overlay. It then renders each saved gauge reading into the matching
overlay region. The full-frame label shows pressure or `pressure unavailable`;
`render_overlay(..., show_label=False)` adds pivot/needle geometry without a second
label inside the crop. Drawing never repeats inference or pressure reading.

**Checkpoint:** Why is `overlay = packet.image` different from
`overlay = packet.image.copy()`?

## 7. What gets saved?

Read `StreamSession.write` and `finish`.
Each frame produces one strict JSON line; even a frame with no detections keeps
its identity. Select `processors` entries by name: `aruco` or `targets`. Target
data includes class names, source-pixel boxes and serialized gauge readings.
Typed drawing objects and crops stay out of the JSON.

An unexpected branch failure marks the frame partial. A normal ArUco or gauge
rejection retains its reason in that object's record. `run.json` saves the model
path, SHA-256, class names and gauge settings under `metadata.targets` when enabled.
On shutdown it records complete/interrupted/failed status and summary statistics.

No image or result history accumulates in memory. Online diagnostics keep counts,
a sum for the mean, first/last arrival samples, and a rolling window of 2048
latencies for P95. The full JSONL remains available for offline statistics.

**Checkpoint:** Why must the reported recent-window P95 be labelled differently
from an all-run percentile?

## 8. How do failures stop?

Read the `try/finally` in `run_stream` and `ExitStack` in `main`.
Before the controller starts, main owns cleanup. It then transfers ownership to
the controller. Camera or display failures still close both resources. Closing
one resource must not prevent the other from closing; a cleanup failure should
not hide the original fault.

A read timeout handles a missing camera message. A processing budget is different:
in this synchronous version it can reject a slow result when it returns, but
cannot cancel a hung model. That distinction is deliberate and documented.

**Checkpoint:** Why would adding `if elapsed > limit` after a function call not
protect us from a function that never returns?

## 9. Where do the results go next?

Native MQTT sends the same frame results, including `targets`. NetGear sends the
annotated image and its identity; the local receiver serves it to the browser.
`run_local_demo.py --camera --targets` forwards the target options to one sender
process and uses that process's interpreter for the model runtime.

The GCS log forwarder samples frames and emits the existing ArUco summary plus
separate gauge and valve messages. Automatic detection-image capture selects
one frame every three seconds while gauges or valves are visible. It saves
untouched full-resolution `frame-N.png` and same-frame `frame-N.annotated.png`
from the existing composition, without another detector pass. All current
targets share one event and one annotated photo upload. ArUco retains its
once-per-sighting rule. GCS retains the annotated copy; originals stay on the Pi.
Both PNGs count toward the default 256 MiB session storage limit, while each
pair uses one of the three pending event slots. The separate uploader observes
its byte budget, so three-second capture selection does not guarantee
three-second delivery over the network.

The v1 event's top-level image descriptor still describes the original; optional
`annotated_image` describes the rendered copy. The uploader prefers that copy
and fails if it is corrupt. Older events without that descriptor upload their
originals, and existing GCS history is not annotated or replayed. See the
[GCS integration guide](gcs-integration.md#automatic-images-aruco-sightings-and-periodic-targets)
for the descriptor fields and upload commands.

Measure target detection quality and combined processing time on the intended
hardware before choosing concurrency, lower processing rates or a hard deadline.
A requested 30 FPS and tests with controlled model output do not establish that
performance.
