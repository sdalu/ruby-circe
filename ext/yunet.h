#ifndef __YUNET__
#define __YUNET__

#include <string>
#include <opencv2/objdetect/face.hpp>
#include <opencv2/core.hpp>

class YuNet
{

public:
    /* Detection thresholds, all within 0..1.
     *  score : lowest confidence kept
     *  nms   : IoU above which two boxes are taken for one face
     */
    struct Threshold {
	float score;
	float nms;

	/* See the note in yolo.h: default member initializers cannot be
	 * used for a default argument inside this same class.
	 */
	Threshold(float s = 0.6f, float n = 0.3f) : score(s), nms(n) {}
    };

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

    void process(const cv::Mat& img, cv::Mat& faces,
		 const Threshold& threshold = Threshold()) {
	model->setScoreThreshold(threshold.score);
	model->setNMSThreshold(threshold.nms);
	model->setInputSize(img.size());
	model->detect(img, faces);
    }

private:

    cv::Ptr<cv::FaceDetectorYN> model;
};

#endif
