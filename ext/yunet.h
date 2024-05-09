#ifndef __YUNET__
#define __YUNET__

#include <string>
#include <opencv2/objdetect/face.hpp>
#include <opencv2/core.hpp>

class YuNet
{

public:
    YuNet(const std::string& model_path,
          const cv::Size&    input_size     = cv::Size(320, 320),
          float              conf_threshold = 0.6f,
          float              nms_threshold  = 0.3f,
          int                top_k          = 5000,
          int                backend_id     = cv::dnn::DNN_BACKEND_OPENCV,
          int                target_id      = cv::dnn::DNN_TARGET_CPU)
    {
        model = cv::FaceDetectorYN::create(model_path, "", input_size,
					   conf_threshold,
					   nms_threshold, top_k,
					   backend_id, target_id);
    }

    ~YuNet() {};

    void process(const cv::Mat& img, cv::Mat& faces) {
	model->setInputSize(img.size());
	model->detect(img, faces);
    }

private:

    cv::Ptr<cv::FaceDetectorYN> model;
};

#endif
