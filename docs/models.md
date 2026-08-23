# Models

Notes on which ONNX exports circe can use, how the bundled ones were
produced, and what was measured. The short version lives in the README;
this is the detail behind it.

## What can be decoded

The post-processing reads the raw detection grid, in either of the two
layouts yolo has used, and infers which by comparing the two output
dimensions:

| Family | Output | Per prediction |
|---|---|---|
| yolov5 | `(1, N, 85)` | 4 box + objectness + 80 scores |
| yolov8, yolo11, yolo26 | `(1, 84, N)` | 4 box + 80 scores |

`N` is the number of candidate boxes and follows from the input size:
6300 at 480x640, 8400 at 640x640. Only the 84 (or 85) is fixed, and it
is what gets checked. yolo26 needs `end2end=False` to produce this at
all; the others do so by default.

Two things are required beyond the shape:

- **80 COCO classes.** The vocabulary is compiled into `ext/yolo.h` and
  the width of a prediction is checked against it. A model trained on a
  different class count is refused rather than misread.
- **A static input shape**, matching `ONNX_YOLO` in `lib/circe.rb`.

An end-to-end export — `(1, 300, 6)`, one finished detection per row — is
refused. See the last section for why that refusal is not just a missing
decoder.

## Bundled

| File | Role |
|---|---|
| `yolo11n.onnx` | default |
| `yolo26n.onnx` | for comparison, not a recommendation |
| `face_detection_yunet_2023mar.onnx` | face detection |

Only the default and the face model are packaged in the gem; `yolo26n`
stays in the repository so `rake bench:compare` can find it.

Switching is the name in `ONNX_YOLO`:

~~~ruby
ONNX_YOLO = [ File.join(DATA_DIR, 'yolo26n.onnx'), 480, 640 ]
~~~

### How they were produced

`yolo11n.onnx` is the published release asset, frozen to a static shape:

~~~sh
curl -LO https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx
# sha256 634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b
onnxslim yolo11n.onnx data/yolo11n.onnx --input-shapes images:1,3,480,640
~~~

`yolo26n.onnx` needs a real export, to get the raw head rather than the
end-to-end one (ultralytics 8.4.127, sha256 `27f517d4…`):

~~~sh
yolo export model=yolo26n.pt format=onnx imgsz=480,640 end2end=False
~~~

## Freezing, and why it is needed

The published yolo11 assets carry dynamic axes — `['batch', 3, 'height',
'width']`. Opencv's ONNX importer cannot fold those and fails on a
Concat in `model.12`, which looks like "opencv does not support yolo11"
but is only about the shape being unknown at import. Freezing the input
resolves it, and drops the shape machinery from the graph (Concat 39 to
28, every `Shape` and `Gather` gone).

A model that is *already* static cannot be reshaped this way: onnxslim
will produce a file whose attention blocks then fail a Reshape
assertion at load. A different input size needs a fresh `yolo export`
with the wanted `imgsz`, which means torch and ultralytics.

Square sizes take a different path in the pre-processing: the image is
letterboxed rather than squashed, preserving aspect ratio.

## Measured

Published accuracy against latency on a raspberry pi 4 (Cortex-A72), all
at 480x640, minimum of five runs, classification only:

| Model | mAP50-95 | pi 4 | peak RSS | file |
|---|---|---|---|---|
| yolo11n | 39.5 | 580 ms | 199 MB | 10.1 MB |
| yolo26n | 40.9 \* | 604 ms | 188 MB | 9.4 MB |
| yolo11s | 47.0 | 1519 ms | 323 MB | 36.2 MB |
| yolo26s | 48.6 \* | 1507 ms | 315 MB | 36.5 MB |
| yolo11m | 51.5 | 4347 ms | 516 MB | 76.9 MB |

\* published for the end-to-end head, which is not the head that can be
run here — see below. yolo26n has the higher figure of the two nano
models and was consistently the weaker of them in practice, finding 3
detections where yolo11n found 8 on the same frame, and 4 horses where
it found 5.

Two notes on reading this. The measurements were taken on a machine that
had a compile running, so they are upper bounds; the ordering is what
matters. And accuracy here is the published mAP, not something measured
locally — a handful of sample images can show a model is wrong, but not
that one is better.

## The yolo26 end-to-end problem

yolo26 is NMS-free: the selection happens in the graph, using `TopK`, and
the default export emits `(1, 300, 6)`. That shape alone would only mean
a decoder was missing. It is worse than that — **opencv 4.13 imports the
model and computes it wrongly**.

Given the same input tensor, byte for byte, onnxruntime returns what the
sample image contains:

~~~
528.113  233.664  639.718  522.002   conf=0.959346  cls=0   person
  6.080  135.772  635.634  439.064   conf=0.959219  cls=5   bus
 40.810  236.698  194.275  535.750   conf=0.947603  cls=0   person
177.033  241.068  272.704  510.724   conf=0.895539  cls=0   person
~~~

while opencv returns the first row's confidence and class repeated across
all three hundred rows, with mis-gathered boxes. It gets row 0's `x1` and
`conf` right and then diverges, which is consistent with the gather that
reorders by the `TopK` indices not being applied per row.

What was ruled out, in order:

- **Not the model, nor the pre-processing.** The comparison above used
  the identical input tensor, dumped from opencv itself.
- **Not opencv's ONNX support in general.** The same comparison on
  yolo11 agrees to 2e-6 on average across 705,600 elements.
- **Not `TopK`.** Tested on its own against numpy, opencv returns
  `[0.95, 0.9, 0.7]` at indices `[5, 1, 3]` — correct.

Reported upstream as
[ultralytics#23524](https://github.com/ultralytics/ultralytics/issues/23524),
closed stale, no fix. The reporter there saw the same thing against
LibTorch on opencv 4.12.

The way round it is `end2end=False`, which exports the raw head instead.
That head *is* computed correctly — checked against onnxruntime, same
2e-6 — and needs no change to circe. But it gives up the reason to want
yolo26: the speed comes from the NMS-free head, and pairing the raw head
with classic NMS is not what the model was trained for. Measured, it is
about a quarter slower than yolo11n and finds fewer objects at any
threshold.
