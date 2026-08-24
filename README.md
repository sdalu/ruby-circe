# circe

Face and object recognition for ruby, over openCV: YuNet for faces,
YOLO11 trained on the 80 COCO classes for objects.

One call decodes an image, runs either or both detectors, and optionally
hands each detection to a block that says how it should be drawn.

~~~text
                            ┌─────────────┐
                       ┌───▸│ YOLO        │───▸ :class, box, name, conf
   bytes ──▸ decode ───┤    └─────────────┘
             3 chan    │    ┌─────────────┐
                       └───▸│ YuNet       │───▸ :face, box, landmark, conf
                            └─────────────┘
~~~


# Install

Ruby 3.4 or later, and openCV 4 with the `dnn`, `objdetect`, `imgproc`
and `imgcodecs` modules, including development files. On FreeBSD that is
`pkg install opencv`, on Debian `apt install libopencv-dev`.

~~~sh
gem install circe
~~~

The extension is built at install time, so openCV must be found by
`pkgconf`/`pkg-config` first. From a checkout instead:

~~~sh
rake compile        # build the extension in place
rake test           # fetches a sample image on first run
rake bench          # see Benchmark
~~~


# Example

~~~ruby
require 'circe'

circe = Circe::new

img = File.read('foo.jpg')
features, out = circe.analyze(img, :jpg) do |type, box, *args, confidence|
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


# What comes back

~~~text
   features, image = circe.analyze(bytes, :jpg) { ... }
   │         │
   │         └──▸ String, the encoded image (nil without a format)
   └────────────▸ Array of [ type, box, ..., confidence ]
~~~

A feature is one of two shapes:

~~~
:class, box, name,     confidence
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

Note the landmark order: the right eye comes first, and it is the
subject's right, so on a frontal face it sits at the smaller x.


# Options

`analyze` takes the image as a byte string, an optional output format
(`:png`, `:jpg`, or `nil` for no image), and:

| Option | | |
|---|---|---|
| `face:` | | run face detection |
| `classify:` | | run object classification |
| `debug:` | `false` | overlay the inference time |
| `threshold:` | | detection thresholds, see below |
| `face_input:` | `640` | longest side the face detector sees |

Selection is a property of the pair, not of either one. Given neither,
both run. Given either, each is taken literally — `true` selects and
`false` refuses — so asking for one turns the other off:

~~~ruby
circe.analyze(img)                    # both
circe.analyze(img, face:     true)    # faces only
circe.analyze(img, classify: false)   # faces only, said the other way
circe.analyze(img, face: true, classify: true)   # both, explicitly
circe.analyze(img, face: false, classify: false) # neither, no features
~~~

Thresholds are all within `0..1`, and any subset may be given. An
unknown key raises rather than being ignored.

| Key | Default | |
|---|---|---|
| `:score` | `0.50` | lowest class score kept |
| `:nms` | `0.50` | IoU above which two boxes are one object |
| `:confidence` | `0.25` | objectness — yolov5 layout only, inert otherwise |
| `:face` | `0.60` | lowest face confidence kept |
| `:face_nms` | `0.30` | IoU above which two boxes are one face |

`:score` is the one that gates the `confidence` value yielded to your
block. `:confidence` is a different thing — objectness — and does
nothing with the bundled model, which has no such term; it is there for
a yolov5-layout model.

## face_input

Classification always resizes to the model's input, so its cost is the
same whatever you feed it. Face detection does not: it runs at the size
of the image it is given, and so gets steadily more expensive as the
source grows. Measured on a raspberry pi 4, the same scene:

| source width | face | classify |
|---|---|---|
| 320 | 23 ms | 654 ms |
| 800 | 127 ms | 519 ms |
| 1280 | 360 ms | 557 ms |
| 1920 | **933 ms** | 691 ms |

At 1080p faces cost more than everything else together. So by default
the image is shrunk to 640 on its longest side for the face pass only,
and the coordinates are scaled back; below that nothing happens.

The cost is the smallest faces. On a 1920x1080 frame, capping at 640
keeps faces down to about 30 px wide and loses them below that. Raise
it, or switch it off, if you need faces further away:

~~~ruby
circe.analyze(img, face_input: 1280)  # shrink less, keep smaller faces
circe.analyze(img, face_input: nil)   # full image, as before
~~~

~~~ruby
# Catch people further away, at the cost of more false positives
features, = circe.analyze(img, threshold: { score: 0.35 })
~~~


# Classes

The 80 COCO names, exactly as carried in the model metadata.
`Circe::CLASSES` holds them.

A name that is not in that list simply never matches — nothing raises,
and the filter quietly yields nothing. So check the list rather than
trusting it:

~~~ruby
VEHICLE = %w[bicycle car motorcycle bus train truck boat].freeze
ANIMAL  = %w[bird cat dog horse sheep cow].freeze
raise "typo" unless (VEHICLE + ANIMAL).all? { Circe::CLASSES.include?(it) }
~~~

Four names changed with the move to yolo11, having previously used the
older VOC spellings. A filter written against a circe before that, or
copied from a yolov5-era source, will match nothing for these:

| was | is now |
|---|---|
| `motorbike` | `motorcycle` |
| `aeroplane` | `airplane` |
| `sofa` | `couch` |
| `tvmonitor` | `tv` |


# Annotating

When a format is given and a block is passed, each feature is yielded to
it. What the block returns says how that detection is drawn:

| Returned | Effect |
|---|---|
| `nil` | nothing drawn for this detection |
| `"label"` | default box, with that label |
| `[ label, color, thickness ]` | trailing entries may be omitted |
| `{ label:, color:, thickness:, extra: }` | same, by name |

`color` is `0xRRGGBB` — spelled the American way, as the keyword.
Within an annotation, `color: nil` draws nothing at all and
`thickness: nil` draws the label without a box. `extra:` draws the five
landmarks, defaults to true, and is ignored for a `:class` feature.

Without a format there is no image to draw on, and the block is not
called at all.


# Models

`yolo11n` at 480x640 is the default, and is bundled. Any yolo ONNX export
works if it produces the raw detection grid over the 80 COCO classes at a
static input size: yolov5u, yolov8 and yolo11 do so by default, and
yolo26 does when exported with `end2end=False`. An end-to-end export —
yolo26's default — is refused.

The model and its input size are set by `ONNX_YOLO` in `lib/circe.rb`,
where `DATA_DIR` is the `data/` directory beside it. There is currently
no runtime setting for this: choosing a model means editing that
constant, which is a from-checkout operation rather than something an
installed gem supports. The extension does not need rebuilding
afterwards — the model is read at `require` time, not compiled in.

The size must be what the model was exported with. Mind the order — it
is **height then width**, matching the ONNX tensor:

~~~ruby
ONNX_YOLO = [ File.join(DATA_DIR, 'yolo11n.onnx'), 480, 640 ]
#                                                  ^^^  ^^^
#                                               height  width
~~~

The benchmark takes `SIZE=` and prints sizes the other way round, as
width x height, so the same model reads as `480,640` here and
`640x480` there.

A square size takes a different path: the image is letterboxed, keeping
its aspect ratio, rather than squashed to fit. The bundled 480x640
squashes.

[docs/models.md](docs/models.md) covers the rest: how the bundled models
were produced, why a dynamic export has to be frozen first, a measured
comparison of the candidates, and why yolo26's default export cannot be
used.


# Benchmark

`rake bench` reports startup, the cost of each detector, and what
encoding an annotated image adds, as min/avg/max over N rounds. Compare
the minimum: it is the one least polluted by whatever else the machine
is doing, and a wide spread is the sign that something else was.

~~~sh
rake bench
rake bench N=20 IMAGE=frame.jpg
rake bench MODEL=data/yolo11n.onnx SIZE=640x480
~~~

`rake bench:compare` puts several side by side, one process each since
the network is built when the extension is required. With no `MODELS` it
compares whatever is in `data/`, so dropping a candidate there is
enough. A model that will not run says why rather than disappearing.

~~~
  model                    load    cold classify   face   both  rss features
  yolo11n @640x480          519     210      148     12    179  220        6
  yolo26n @640x480          626     174      131      8    158  212        3
  yolo26s @640x640     yolo: cannot decode a model giving 6 values per ...
~~~

For reference, a raspberry pi 4 running the bundled model: about 580 ms
to classify a frame, 60 ms for faces, in 200 MB.


# See

* [YuNet](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet)
  — the face detector, and where its ONNX comes from
* [libfacedetection](https://github.com/ShiqiYu/libfacedetection) — the
  work YuNet grew out of
* [YOLO11](https://docs.ultralytics.com/models/yolo11/) — the object
  model, its accuracy figures and export options

# Credit

* `ext/camera_model.h`, from
  [iwatake2222](https://github.com/iwatake2222), for the head-pose
  estimation that is currently kept but not built
