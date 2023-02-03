# Intro

Based on model from YuNet for face detaction, and YOLO for object
classification.


# Types

~~~
:class, box, name, confidence
:face,  box, landmark

box        : [ x: Integer, y: Integer, width: Integer, height: Integer ]
confidence : Integer
landmark   : [ left_eye  : [ x: Integer, y: Integer ],
               right_eye : [ x: Integer, y: Integer ],
               nose      : [ x: Integer, y: Integer ],
               left_lip  : [ x: Integer, y: Integer ],
               right_lip : [ x: Integer, y: Integer ] ]
~~~

# Example

~~~ruby
require 'circe'

$circe = Circe::new

img = File.read('foo.jpg')
features, out = $circe.analyze(img, :jpg) do |type, box, *args|
    case type
    when :class
        name, confidence = args
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
