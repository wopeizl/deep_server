#pragma once

#include "caffe_config.hpp"
#include "deep_server.hpp"
using namespace caffe;

//inline std::string INT(float x) { char A[100]; sprintf(A, "%.1f", x); return std::string(A); };
inline std::string FloatToString(float x) { char A[100]; sprintf(A, "%.4f", x); return std::string(A); };

class caffe_process {
public:
    bool init(vector<string>& flags, int gpu_id);

    bool prepare(const cv::Mat &input, cv::Mat &img);
    bool preprocess(const cv::Mat &img_in, vector<boost::shared_ptr<Blob<float> >>& input);
    //bool predict(vector<boost::shared_ptr<Blob<float> >>& input, vector<boost::shared_ptr<Blob<float> >>& output);
    bool postprocess(const cv::Mat &img, vector<boost::shared_ptr<Blob<float> >>& input, std::vector<caffe::Frcnn::BBox<float> > &output);
    bool predict_b(const std::vector<cv::Mat >& img_in, vector<boost::shared_ptr<Blob<float> > >& output);
    bool postprocess_b(const std::vector<cv::Mat >& img_in, int imat, vector<boost::shared_ptr<Blob<float> > >& input, std::vector<caffe::Frcnn::BBox<float> > &results);

    caffe_process() {
        DEEP_LOG_INFO("new caffe process!");
    }
    ~caffe_process() {}
private:
    FRCNN_API::Frcnn_wrapper* wrapper;
    static bool initialized;
};

namespace deep_server {

    struct caffe_data {
        actor self;
        cv::Mat cv_image;
        cv::Mat out_image;
        std::vector<boost::shared_ptr<caffe::Blob<float> >> input;
        std::vector<boost::shared_ptr<caffe::Blob<float> >> output;
        std::vector<caffe::Frcnn::BBox<float>> results;
        connection_handle handle;
        int resDataType;
        struct time_consume time_consumed;
        std::string callback;
        tcp_output t_out;
        http_output h_out;
    };

    class caffe_processor : public event_based_actor {
    public:

        ~caffe_processor() {
            DEEP_LOG_INFO("Whole time for caffe to process this image : " + boost::lexical_cast<string>(pcaffe_data->time_consumed.whole_time) + " ms!");
            DEEP_LOG_INFO("Time details  : "
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.jsonparse_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.decode_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.cvreadimage_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.prepare_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.preprocess_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.predict_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.postprocess_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.writeresult_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.whole_time) + ","
                + " ms!");

            DEEP_LOG_INFO("Finish this round processor.");
        }

        caffe_processor(actor_config& cfg,
            std::string  n, actor s, connection_handle handle, actor cf_manager);

    protected:
        behavior make_behavior() override;

        void fault(std::string reason) {
            send(bk, handle_, fault_atom::value, reason);
        }

    private:
        actor bk;
        actor cf_manager;
        connection_handle handle_;
        std::string name_;
        behavior    input_;
        behavior    framesync_;
        behavior    prepare_;
        behavior    preprocess_;
        behavior    downloader_;
        behavior    processor_;
        std::unique_ptr<caffe_data> pcaffe_data;
        bool http_mode;
    };
}