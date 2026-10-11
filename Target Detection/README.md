### Pixi Usage

- pixi install
- pixi run python3 file.py

### Downloading dataset (for PyTests)
- pixi run python assets/download_dataset.py

### PyTest Usage

- Note: Before doing this you must have downloaded the dataset

- pixi run pytest -q tests/<file_name>

- pixi run pytest -q -rP tests/<file_name> -- for logging prints to terminal

### PyTest Implementation

- classes must begin with Test
- child methods must begin with test

### Integrating with system
- All detections are done in the Predictor class inside src/predict.py
1. from src/predict.py import initialise_predictor
2. get the instance of predictor by calling initialise_predictor()
3. cleaned_prediction = predictor.predict(raw_image)

Note: cleaned_prediction will be in the form [(detection_class, (x_centre, y_centre, width, height))]
where each of the floats are percentages of image

4. latest_gauge_crop = predictor.get_latest_gauge_crop()

Note: this helper returns only the last gauge from the most recent prediction.
It returns `None` before the first gauge, when the next frame contains no gauge,
or when the stored crop is more than 30 seconds old. Use it for a single-crop
preview, not for attaching readings to stream frames.

### Passing detections to the stream

Call `predictor.predict_detections(raw_image)` once for each frame. This runs
the same model and crop calculation as `predict()`, but returns every target
with the information the stream needs:

- `class_name`: gauge, openValve or closedValve.
- `box_xywhn`: the original normalized box for `annotate_image()`.
- `bbox_xyxy_px`: left, top, right and bottom in the original frame's pixels.
- `gauge_crop`: a separate, unannotated image for each gauge; `None` for valves.

The right and bottom edges are excluded, matching NumPy image slicing. An empty
frame returns `[]`, so previous gauges are not attached to a new frame.

`TargetDetectionProcessor` passes each supplied gauge crop to `read_gauge()`.
The reader still performs its own preprocessing and returns either a pressure
or a failure reason. The adapter retains both outcomes with the target box and
frame identity. Image arrays stay out of the saved JSON results.

Use either `predict()` for the existing label-only demos or `predict_detections()`
for the stream. Calling both on the same frame would run the model twice.

`TargetDetectionProcessor.draw()` uses the saved results to call the existing
target and gauge renderers. It draws the target boxes with a pressure value or
`pressure unavailable`, then adds any available dial and needle geometry. It does
not run prediction or reading again. Drawing uses the accumulated display copy,
so the original image and earlier annotations are preserved.

The processor now supplies both `process()` and `draw()` for the shared stream.
Registration in the command-line runner and target-specific GCS forwarding remain
separate integration steps.

### Checking the frame handoff

These checks use controlled model outputs, so they do not need a camera, downloaded
weights or image windows:

```sh
pixi run python -B -m pytest -q tests/test_predictor_frames.py
```
