#pragma once

#include "deep_server.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#include "image.h"
#if defined WIN32 || defined WINCE
#else
#include "yolo_wrapper.h"
#endif

#ifdef __cplusplus
}
#endif


class yolo_process {
public:

    bool init(int gpu_id, const char *cfgfile
        , const char *weightfile
        , const char* namelist
        , const char* labeldir
        , float thresh
        , float hier_thresh
    );

#if defined WIN32 || defined WINCE
#else

    static bool prepare(const cv::Mat &img, image* output);

    bool predict(image* input, cv::Mat& output);
#endif

    static bool postprocess(image input, cv::Mat &output);

    yolo_process() /*: wrapper(nullptr)*/{
        DEEP_LOG_INFO("new yolo process!");
    }
    ~yolo_process() {}

private:
    static bool initialized;
    //yolo_wrapper* wrapper;
};

namespace deep_server {
    struct yolo_data {
        actor self;
        cv::Mat cv_image;

        image input;
        cv::Mat output;
        connection_handle handle;
        struct time_consume time_consumed;
    };

    class yolo_processor : public event_based_actor {
    public:

        ~yolo_processor() {
            DEEP_LOG_INFO("Whole time for yolo to process this image : " + boost::lexical_cast<string>(pyolo_data->time_consumed.whole_time) + " ms!");
            DEEP_LOG_INFO("Time details  : "
                + boost::lexical_cast<string>(pyolo_data->time_consumed.jsonparse_time) + ","
                + boost::lexical_cast<string>(pyolo_data->time_consumed.decode_time) + ","
                + boost::lexical_cast<string>(pyolo_data->time_consumed.cvreadimage_time) + ","
                + boost::lexical_cast<string>(pyolo_data->time_consumed.prepare_time) + ","
                + boost::lexical_cast<string>(pyolo_data->time_consumed.preprocess_time) + ","
                + boost::lexical_cast<string>(pyolo_data->time_consumed.predict_time) + ","
                + boost::lexical_cast<string>(pyolo_data->time_consumed.postprocess_time) + ","
                + boost::lexical_cast<string>(pyolo_data->time_consumed.writeresult_time) + ","
                + boost::lexical_cast<string>(pyolo_data->time_consumed.whole_time) + ","
                + " ms!");

            DEEP_LOG_INFO("Finish this round processor.");
        }

        yolo_processor(actor_config& cfg,
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
        std::unique_ptr<yolo_data> pyolo_data;
    };
}