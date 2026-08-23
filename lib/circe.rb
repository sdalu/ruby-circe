class Circe

    private
    
    # Don't know how to do it inside the c extension
    DATA_DIR     = File.join(__dir__, '..', 'data').freeze

    # Model, and the input size it was exported with. Also bundled is
    # yolo26n.onnx, same size and vocabulary, usable by swapping the name
    # below -- but measurably slower here, and finding less. See the
    # README, it is kept for comparison rather than as a recommendation.
    ONNX_YOLO     = [ File.join(DATA_DIR, 'yolo11n.onnx'), 480, 640 ]
    ONNX_YUNET    = [ File.join(DATA_DIR, 'face_detection_yunet_2023mar.onnx') ]
    
end

require_relative 'circe/version'
require_relative 'circe/core'
