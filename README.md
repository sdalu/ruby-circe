# Intro

Based on model from YuNet for face detection, and YOLO11 with COCO training
for object classification.


# Types

~~~
:class, box, name, confidence
:face,  box, landmark, confidence

name       : String
box        : [ x: Integer, y: Integer, width: Integer, height: Integer ]
confidence : Float
landmark   : [ right_eye          : [ x: Integer, y: Integer ],
               left_eye           : [ x: Integer, y: Integer ],
               nose_tip           : [ x: Integer, y: Integer ],
               right_corner_mouth : [ x: Integer, y: Integer ],
               left_corner_mouth  : [ x: Integer, y: Integer ] ]
~~~

# Options

`analyze` takes the image as a byte string, an optional output format
(`:png`, `:jpg`, or `nil` for no image), and:

| Option | Default | |
|---|---|---|
| `face:` | | run face detection |
| `classify:` | | run object classification |
| `debug:` | `false` | overlay the inference time |
| `threshold:` | | detection thresholds, see below |

Neither `face:` nor `classify:` given runs both. Giving one as `true` runs
only that one; giving one as `false` runs the other.

Thresholds are all within `0..1`, and any subset may be given:

| Key | Default | |
|---|---|---|
| `:score` | `0.50` | lowest class score kept |
| `:nms` | `0.50` | IoU above which two boxes are one object |
| `:confidence` | `0.25` | objectness, only used by the yolov5 layout |
| `:face` | `0.60` | lowest face confidence kept |
| `:face_nms` | `0.30` | IoU above which two boxes are one face |

~~~ruby
# Catch people further away, at the cost of more false positives
features, = circe.analyze(img, threshold: { score: 0.35 })
~~~

Class names are the 80 modern COCO ones, exactly as carried in the model
metadata. `Circe::CLASSES` holds them, so a filter list can be checked
rather than guessed at:

~~~ruby
VEHICLE = %w[bicycle car motorcycle bus train truck boat].freeze
ANIMAL  = %w[bird cat dog horse sheep cow].freeze
raise "typo" unless (VEHICLE + ANIMAL).all? { Circe::CLASSES.include?(it) }
~~~

# Models

Any yolo ONNX export whose output is the raw `(1, 4 + classes, boxes)` or
`(1, boxes, 5 + classes)` tensor, trained on the 80 COCO classes. That
covers yolov5u, yolov8 and yolo11. It does *not* cover an end-to-end or
`nms=True` export, whose `(1, 300, 6)` output is refused.

The path and input size are set by `ONNX_YOLO` in `lib/circe.rb`, and the
size must match what the model was exported with. A square size makes the
image letterboxed rather than squashed.

The bundled `data/yolo11n.onnx` is the ultralytics release asset, frozen
to a static 480x640 input:

~~~sh
curl -LO https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx
# sha256 634279b40c07c6391472c51ad45b81ebc48706a9a1fe72dd3396322acd0c053b
onnxslim yolo11n.onnx data/yolo11n.onnx --input-shapes images:1,3,480,640
~~~

Freezing is needed because opencv's ONNX importer cannot fold the dynamic
shape nodes that release carries, and fails on a Concat. The other size
must come from a release exported at it, or from a fresh `yolo export`:
reshaping an already static model breaks its attention blocks.

# Example

~~~ruby
require 'circe'

$circe = Circe::new

img = File.read('foo.jpg')
features, out = $circe.analyze(img, :jpg) do |type, box, *args, confidence|
    case type
    when :class
        name, = args
        next nil unless [ 'person' ].include?(name)
        [ "%s: %.2f" % [ name, confidence ], 0xff00f0, 5 ]
    when :face
        "face"
    end
end

File.write('foo-annotated.jpg', out)
~~~

# See

* https://github.com/ShiqiYu/libfacedetection
* https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
* https://github.com/ultralytics/yolov5/releases

# Credit

* `camera_model.h` from iwatake2222
