class Circe

    private
    
    # Don't know how to do it inside the c extension
    DATA_DIR     = File.join(__dir__, '..', 'data').freeze
    ONNX_YOLO     = [ File.join(DATA_DIR, 'yolov8s.onnx'), 480, 640 ]
    ONNX_YUNET    = [ File.join(DATA_DIR, 'face_detection_yunet_2023mar.onnx') ]
    
end

require_relative 'circe/version'
require_relative 'circe/core'
