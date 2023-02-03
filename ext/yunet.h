#ifndef __YUNET__
#define __YUNET__

#include <string>
#include <vector>
#include <array>
#include <utility>

#include <opencv2/dnn.hpp>


class YuNet
{

public:
    typedef std::array<cv::Point, 5> Landmark;
    typedef std::pair<cv::Rect, Landmark> Face;
    
private:
    static constexpr int                MODEL_WIDTH    = 512;
    static constexpr float              CONF_THRESHOLD = 0.4f;
    static constexpr float              NMS_THRESHOLD  = 0.3f;

    const std::vector<float>            VARIANCES      = { 0.1f, 0.2f };
    const std::vector<int>              STEPS          = { 8, 16, 32, 64 };
    const std::vector<std::vector<int>> MIN_SIZES      = {
	{ 10, 16, 24 }, { 32, 48 }, { 64, 96 }, { 128, 192, 256 } };

public:
    YuNet(const std::string& model);
    ~YuNet() {};
    void process(const cv::Mat& img, std::vector<Face>& faces);

private:
    cv::dnn::Net net;
};

#endif
