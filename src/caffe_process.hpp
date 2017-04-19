#pragma once

#include "caffe_config.hpp"
using namespace caffe;

//inline std::string INT(float x) { char A[100]; sprintf(A, "%.1f", x); return std::string(A); };
inline std::string FloatToString(float x) { char A[100]; sprintf(A, "%.4f", x); return std::string(A); };

class caffe_process {
public:
    bool init(vector<string>& flags);

    bool prepare(const cv::Mat &input, cv::Mat &img);
    bool preprocess(const cv::Mat &img_in, vector<boost::shared_ptr<Blob<float> >>& input);
    bool predict(vector<boost::shared_ptr<Blob<float> >>& input, vector<boost::shared_ptr<Blob<float> >>& output);
    bool postprocess(const cv::Mat &img, vector<boost::shared_ptr<Blob<float> >>& input, std::vector<caffe::Frcnn::BBox<float> > &output);

    caffe_process() {}
    ~caffe_process() {}
private:
    FRCNN_API::Frcnn_wrapper* wrapper;
    static bool initialized;
};