/*
 * https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-CPP-Inference/inference.cpp
 * https://github.com/opencv/opencv/blob/4.x/samples/dnn/yolo_detector.cpp
 *
 * yolov5 has an output of shape:
 *  (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
 * yolov8 has an output of shape:
 *  (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
 *
 * yolo export model=yolov8s.pt imgsz=480,640 format=onnx opset=12
 */ 
#include <tuple>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "yolo.h"


Yolo::Yolo(const std::string& model, cv::Size size) {
    this->net  = cv::dnn::readNetFromONNX(model);
    this->size = size;
}


void Yolo::process(cv::Mat &img, std::vector<Yolo::Item> &items) {
    int version = 5;

    cv::Mat input = img;
    if (letterBoxForSquare && size.width == size.height)
        input = formatToSquare(input);
    
    // Pre-process
    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob, 1./255.,
			   this->size, cv::Scalar(), true, false);

    // Process
    std::vector<cv::Mat> outputs;
    net.setInput(blob);
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Output
    int    rows       = outputs[0].size[1];
    int    dimensions = outputs[0].size[2];

    // Infer yolo version
    //  -> Check if the shape[2] is more than shape[1] (yolov8)
    if (dimensions > rows) {
        version = 8;
    }

    // Adjust according to version
    if (version == 8) {
	//  cv::transposeND(outs[0], {0, 2, 1}, outs[0]);
        rows       = outputs[0].size[2];
        dimensions = outputs[0].size[1];
        outputs[0] = outputs[0].reshape(1, dimensions);
        cv::transpose(outputs[0], outputs[0]);
    }

    // Output
    float *data    = (float *)outputs[0].data;

    // Resizing factor.
    float x_factor = input.cols / size.width;
    float y_factor = input.rows / size.height;

    // Post-process
    std::vector<int>       class_ids;
    std::vector<float>     confidences;
    std::vector<cv::Rect>  boxes;
    

    if        (version == 5) {
	for (int i = 0; i < rows; ++i) {
            float confidence = data[4];

            if (confidence >= CONFIDENCE_THRESHOLD) {
                float *classes_scores = data + 5;

                cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;

                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > SCORE_THRESHOLD) {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left   = int((x - 0.5 * w) * x_factor);
                    int top    = int((y - 0.5 * h) * y_factor);
                    int width  = int(w * x_factor);
                    int height = int(h * y_factor);

                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
	    data += dimensions;
	}
    } else if (version == 8) {
	for (int i = 0; i < rows; ++i) {
	    float *classes_scores = data + 4;
	    
            cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;

            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > SCORE_THRESHOLD) {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left   = int((x - 0.5 * w) * x_factor);
                int top    = int((y - 0.5 * h) * y_factor);
                int width  = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
	    data += dimensions;
	}
    }
   
    
    // Perform Non-Maximum Suppression and draw predictions.
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences,
		      SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

    items.clear();

    for (int i = 0; i < nms_result.size(); i++) {
        int  idx = nms_result[i];
	items.push_back({ classes[class_ids[idx]],confidences[idx],boxes[idx] });
    }
}


cv::Mat Yolo::formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}
