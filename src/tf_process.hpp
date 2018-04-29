#pragma once

#include "deep_server.hpp"
#include "tf_wrapper.h"

//#if defined(WIN32) || defined(_WIN32)
//#else
//struct p_tf_detect_result {
//        int nclass;
//        float score;
//        float x, y, w, h;
//    };
//#endif

class tf_process {
public:
    bool init(vector<string>& flags, int gpu_id);
    //bool init(string model_name, string model_base_path, string model_config_file);

    bool prepare();
    bool preprocess();
    bool detect_predict(cv::Mat & img, vector<c_tf_detect_result>& output);
    bool classify_predict(cv::Mat & img, vector<c_tf_classify_result>& output);
    bool postprocess(const std::vector<vector<unsigned char> >& img_in, int imat, 
        vector<c_tf_detect_result>& output, vector<deep_server::Tf_detect_result>& results);
    bool postprocess(const std::vector<vector<unsigned char> >& img_in, int imat, 
        vector<c_tf_classify_result>& output, vector<deep_server::Tf_classify_result>& results);

    tf_process() {
        DEEP_LOG_INFO("new tensorflow process!");
    }
    ~tf_process() {}

private:
    //static bool initialized;
    tf_wrapper::tf_wrapper wrapper;
};

namespace deep_server {

    struct tf_data {
        //actor_control_block* self;
        actor self;
        cv::Mat cv_image;
        cv::Mat out_image;
        std::vector<unsigned char> input;
        connection_handle handle;
        int resDataType;
        std::vector<deep_server::Tf_detect_result> detect_results;
        std::vector<deep_server::Tf_classify_result> classify_results;
        struct time_consume time_consumed;
        std::string callback;
        tcp_output t_out;
        http_output h_out;
    };

    class tf_processor : public event_based_actor {
    public:

        ~tf_processor() {
        }

        void release() {
            DEEP_LOG_INFO("Whole time for caffe to process this image : " + boost::lexical_cast<string>(ptf_data->time_consumed.whole_time) + " ms!");
            DEEP_LOG_INFO("Time details  : "
                + boost::lexical_cast<string>(ptf_data->time_consumed.jsonparse_time) + ","
                + boost::lexical_cast<string>(ptf_data->time_consumed.decode_time) + ","
                + boost::lexical_cast<string>(ptf_data->time_consumed.cvreadimage_time) + ","
                + boost::lexical_cast<string>(ptf_data->time_consumed.prepare_time) + ","
                + boost::lexical_cast<string>(ptf_data->time_consumed.preprocess_time) + ","
                + boost::lexical_cast<string>(ptf_data->time_consumed.predict_time) + ","
                + boost::lexical_cast<string>(ptf_data->time_consumed.postprocess_time) + ","
                + boost::lexical_cast<string>(ptf_data->time_consumed.writeresult_time) + ","
                + boost::lexical_cast<string>(ptf_data->time_consumed.whole_time) + ","
                + " ms!");

            DEEP_LOG_INFO("Finish this round processor.");
            ptf_data.reset(nullptr);
        }

        tf_processor(actor_config& cfg,
            std::string  n, actor s, connection_handle handle, actor cf_manager);

    protected:
        behavior make_behavior() override;

        void fault(std::string reason) {
            send(bk, handle_, fault_atom::value, reason);
            release();
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
        std::unique_ptr<tf_data> ptf_data;
        bool http_mode;
    };
}
