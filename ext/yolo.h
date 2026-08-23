#ifndef __YOLO__
#define __YOLO__

#include <string>
#include <vector>
#include <tuple>

#include <opencv2/dnn.hpp>


class Yolo
{
public:
    typedef std::tuple<std::string, float, cv::Rect> Item;

    /* Detection thresholds, all within 0..1.
     *  confidence : objectness, only present in the v5 layout
     *  score      : lowest class score kept
     *  nms        : IoU above which two boxes are taken for one object
     */
    struct Threshold {
	float confidence;
	float score;
	float nms;

	/* A constructor rather than default member initializers: those
	 * cannot be used for a default argument inside this same class.
	 */
	Threshold(float c = 0.25f, float s = 0.50f, float n = 0.50f)
	    : confidence(c), score(s), nms(n) {}
    };

private:
    static constexpr float INPUT_WIDTH          = 640.0;
    static constexpr float INPUT_HEIGHT         = 640.0;

    bool letterBoxForSquare = true;

 public:
    const std::vector<std::string> classes = {
	"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
	"truck", "boat", "traffic light", "fire hydrant", "stop sign",
	"parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
	"cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
	"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
	"surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
	"knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
	"broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
	"sofa", "potted plant", "bed", "dining table", "toilet", "tvmonitor",
	"laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
	"oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush"
    };

public:
    Yolo(const std::string& model, cv::Size size);
    void process(cv::Mat &img, std::vector<Item> &items,
		 const Threshold& threshold = Threshold());
    
private:
    cv::dnn::Net net;
    cv::Size     size;
    cv::Mat formatToSquare(const cv::Mat &source);
};

#endif
