#pragma once

#include "caffe_config.hpp"

//inline std::string INT(float x) { char A[100]; sprintf(A, "%.1f", x); return std::string(A); };
inline std::string FloatToString(float x) { char A[100]; sprintf(A, "%.4f", x); return std::string(A); };

class caffe_process {
public:
    bool init(vector<string>& flags);

    bool predict(cv::Mat& cv_image, std::vector<caffe::Frcnn::BBox<float>>& results);

    caffe_process() {}
    ~caffe_process() {}
private:
    FRCNN_API::Detector* detector;
};