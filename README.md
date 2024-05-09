# Intro

Based on model from YuNet for face detection, and YOLOv8 with COCO training
for object classification.


# Types

~~~
:class, box, name, confidence
:face,  box, landmark, confidence

name       : String
box        : [ x: Integer, y: Integer, width: Integer, height: Integer ]
confidence : Float
landmark   : [ left_eye           : [ x: Integer, y: Integer ],
               right_eye          : [ x: Integer, y: Integer ],
               nose_tip           : [ x: Integer, y: Integer ],
               left_corner_mouth  : [ x: Integer, y: Integer ],
               right_corner_mouth : [ x: Integer, y: Integer ] ]
~~~

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
