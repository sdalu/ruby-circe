#include <cmath>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>

#include <opencv2/dnn.hpp>

#include "yunet.h"


YuNet::YuNet(const std::string& model_filename)
{
    net = cv::dnn::readNetFromONNX(model_filename);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}


void YuNet::process(const cv::Mat& img, std::vector<YuNet::Face>& faces)
{
    /* -- Preparing for image size -- */
    cv::Size model_size;
    model_size.width  = MODEL_WIDTH;
    model_size.height = MODEL_WIDTH * img.rows / img.cols;
    model_size.height = (model_size.height / 32) * 32;

    std::pair<int32_t, int32_t> feature_map_2th = {
	(model_size.height + 1) / 2 / 2,
	(model_size.width  + 1) / 2 / 2
    };

    std::vector<std::pair<int32_t, int32_t>> feature_map_list;
    feature_map_list.push_back({ (feature_map_2th.first  + 1) / 2 ,
	                         (feature_map_2th.second + 1) / 2 });

    for (int32_t i = 0; i < 3; i++) {
        const auto& previous = feature_map_list.back();
        feature_map_list.push_back({ (previous.first  + 1) / 2 ,
		                     (previous.second + 1) / 2 });
    }

    std::vector<std::vector<float>> prior_list;
    for (int i = 0; i < static_cast<int32_t>(feature_map_list.size()); i++) {
        const auto& min_sizes   = MIN_SIZES[i];
        const auto& feature_map = feature_map_list[i];
        for (int y = 0; y < feature_map.first; y++) {
            for (int x = 0; x < feature_map.second; x++) {
                for (const auto& min_size : min_sizes) {
                    float s_kx = static_cast<float>(min_size) / model_size.width;
                    float s_ky = static_cast<float>(min_size) / model_size.height;
                    float cx = (x + 0.5f) * STEPS[i] / model_size.width;
                    float cy = (y + 0.5f) * STEPS[i] / model_size.height;
                    prior_list.push_back({ cx, cy, s_kx, s_ky });
                }
            }
        }
    }

    
    /* -- Pre-process -- */
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1.0, model_size);
        
    /* -- Inference -- */
    std::vector<cv::Mat> outputs;
    net.setInput(blob);
    net.forward(outputs, { "conf", "iou",  "loc" });

    /* -- Post Process -- */
    const cv::Mat& mat_conf   = outputs[0];
    const cv::Mat& mat_iou    = outputs[1];
    const cv::Mat& mat_loc    = outputs[2];
    const cv::Size image_size = img.size();    

    // Get score list
    std::vector<float> cls_score;
    for (int32_t row = 0; row < mat_conf.rows; row++) {
        float val = mat_conf.at<float>(cv::Point(1, row));
        cls_score.push_back(std::clamp(val, 0.0f, 1.0f));
    }

    std::vector<float> iou_score;
    for (int32_t row = 0; row < mat_iou.rows; row++) {
        float val = mat_conf.at<float>(cv::Point(0, row));
        iou_score.push_back(std::clamp(val, 0.0f, 1.0f));
    }

    std::vector<float> score;
    for (int32_t row = 0; row < mat_conf.rows; row++) {
        score.push_back(std::sqrt(cls_score[row] * iou_score[row]));
    }

    // All bbox
    std::vector<cv::Rect> bbox_all;
    for (int row = 0; row < mat_loc.rows; row++) {
        float cx = mat_loc.at<float>(cv::Point(0, row));
        float cy = mat_loc.at<float>(cv::Point(1, row));
        float w  = mat_loc.at<float>(cv::Point(2, row));
        float h  = mat_loc.at<float>(cv::Point(3, row));

        cx = prior_list[row][0] + cx * VARIANCES[0] * prior_list[row][2];
        cy = prior_list[row][1] + cy * VARIANCES[0] * prior_list[row][3];
        w  = prior_list[row][2] * std::exp(w * VARIANCES[0]);
        h  = prior_list[row][3] * std::exp(h * VARIANCES[1]);

        bbox_all.push_back({
		static_cast<int32_t>((cx - w / 2) * image_size.width),
		static_cast<int32_t>((cy - h / 2) * image_size.height),
		static_cast<int32_t>(w * image_size.width),
		static_cast<int32_t>(h * image_size.height) });
    }

    // Non-Maximum Suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bbox_all, score, CONF_THRESHOLD, NMS_THRESHOLD, indices);

    // Get valid bbox and landmark
    faces.clear();
    for (int idx : indices) {
        Landmark landmark; // (landmark is 5 points)
        for (int i = 0; i < static_cast<int>(landmark.size()); i++) {
	    cv::Point& p = landmark[i];
            float x      = mat_loc.at<float>(cv::Point(4 + i * 2,     idx));
            float y      = mat_loc.at<float>(cv::Point(4 + i * 2 + 1, idx));
            p.x = static_cast<int32_t>((prior_list[idx][0] + x * VARIANCES[0] * prior_list[idx][2]) * image_size.width);
            p.y = static_cast<int32_t>((prior_list[idx][1] + y * VARIANCES[0] * prior_list[idx][3]) * image_size.height);
        }

	faces.push_back({ bbox_all[idx], landmark });
    }
}
