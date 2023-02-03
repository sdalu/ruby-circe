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
    static constexpr float SCORE_THRESHOLD      =   0.5;
    static constexpr float NMS_THRESHOLD        =   0.45;
    static constexpr float CONFIDENCE_THRESHOLD =   0.45;

 public:
    const std::vector<std::string> names = {
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
    Yolo(const std::string& model);
    void process(cv::Mat &img, std::vector<Item> &items);
    
private:
    cv::dnn::Net net;
};

#endif
