#pragma once

#if defined(WIN32) || defined(_WIN32)
#define HAVE_STRUCT_TIMESPEC
#endif

namespace yolo {
    class yolo_wrapper {
    public:
        yolo_wrapper() {}
        ~yolo_wrapper() {}

        bool yolo_init_net_cpp(int gpu_id, const char *cfgfile
            , const char *weightfile
            , const char* namelist
            , const char* labeldir
            , float thresh
            , float hier_thresh);

        bool yolo_predict_cpp(void* input, void* output);
    };
}