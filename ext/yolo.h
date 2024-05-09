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
    
private:
    static constexpr float INPUT_WIDTH          = 640.0;
    static constexpr float INPUT_HEIGHT         = 640.0;
    static constexpr float CONFIDENCE_THRESHOLD =   0.25;
    static constexpr float SCORE_THRESHOLD      =   0.50;
    static constexpr float NMS_THRESHOLD        =   0.50;

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
    void process(cv::Mat &img, std::vector<Item> &items);
    
private:
    cv::dnn::Net net;
    cv::Size     size;
    cv::Mat formatToSquare(const cv::Mat &source);
};

#endif
