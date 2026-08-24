#ifndef __YUNET__
#define __YUNET__

#include <string>
#include <algorithm>
#include <opencv2/objdetect/face.hpp>
#include <opencv2/imgproc.hpp>
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

    /* The detector is run at the size of the image it is given, so its
     * cost follows the source resolution instead of being fixed the way
     * a yolo pass is: on a raspberry pi 4 the same scene costs 23 ms at
     * 320 px wide and 933 ms at 1920. Beyond `limit` the image is
     * shrunk for detection and the coordinates scaled back, which trades
     * the smallest faces for that time. A limit of 0 disables it.
     */
    void process(const cv::Mat& img, cv::Mat& faces,
		 const Threshold& threshold = Threshold(),
		 int limit = 0) {
	model->setScoreThreshold(threshold.score);
	model->setNMSThreshold(threshold.nms);

	int longest = std::max(img.cols, img.rows);
	if ((limit <= 0) || (longest <= limit)) {
	    model->setInputSize(img.size());
	    model->detect(img, faces);
	    return;
	}

	double  scale = (double)limit / longest;
	cv::Mat shrunk;
	cv::resize(img, shrunk, cv::Size(), scale, scale, cv::INTER_AREA);
	model->setInputSize(shrunk.size());
	model->detect(shrunk, faces);

	/* Columns 0..13 are the box then the five landmarks, 14 is the
	 * confidence and must be left alone.
	 */
	float inv = (float)(1.0 / scale);
	for (int i = 0; i < faces.rows; i++)
	    for (int j = 0; (j < 14) && (j < faces.cols); j++)
		faces.at<float>(i, j) *= inv;
    }

private:

    cv::Ptr<cv::FaceDetectorYN> model;
};

#endif
