class Circe

    private
    
    # Don't know how to do it inside the c extension
    DATA_DIR   = File.join(__dir__, '..', 'data').freeze
    ONNX_YOLO  = File.join(DATA_DIR, 'yolov5s.onnx')
    ONNX_YUNET = File.join(DATA_DIR, 'face_detection_yunet_2022mar.onnx')
    
end

require_relative 'circe/version'
require_relative 'circe/core'
