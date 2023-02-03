#include <tuple>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "yolo.h"


Yolo::Yolo(const std::string& model) {
    net = cv::dnn::readNetFromONNX(model);
}


void Yolo::process(cv::Mat &img, std::vector<Yolo::Item> &items) {

    // Pre-process
    cv::Mat blob;
    std::vector<cv::Mat> outputs;

    cv::dnn::blobFromImage(img, blob, 1./255.,
			   cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
			   cv::Scalar(), true, false);

    // Process
    net.setInput(blob);
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Post-process
    std::vector<int>   class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect>  boxes;

    // Resizing factor.
    float x_factor = img.cols / INPUT_WIDTH;
    float y_factor = img.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;
    const int dimensions = 85;
    
    // 25200 for default size 640.
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i) {
	float confidence = data[4];

	// Discard bad detections and continue.
	if (confidence >= CONFIDENCE_THRESHOLD) {
            float *classes_scores = data + 5;

            // Create a 1x85 Mat and store class scores of 80 classes.
	    cv::Mat scores(1, names.size(), CV_32FC1, classes_scores);

            // Perform minMaxLoc and acquire the index of best class score.
	    cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD) {
                // Store class ID and confidence in the pre-defined
		// respective vectors.
                float cx = data[0];  // Center
                float cy = data[1];
                float w  = data[2];  // Box dimension
                float h  = data[3];
		
                // Bounding box coordinates.
                int left   = int((cx - 0.5 * w) * x_factor);
                int top    = int((cy - 0.5 * h) * y_factor);
                int width  = int(w              * x_factor);
                int height = int(h              * y_factor);

                // Store good detections in the boxes vector.
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        // Jump to the next row.
        data += 85;
    }
    
    // Perform Non-Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences,
		      SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    items.clear();

    for (int i = 0; i < indices.size(); i++) {
        int  idx = indices[i];
	items.push_back({ names[class_ids[idx]],confidences[idx],boxes[idx] });
    }
}
