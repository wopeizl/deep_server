
#include "yolo_wrapper.hpp"

#ifdef USE_CUDA
#include "cuda_runtime.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "option_list.h"
#include "blas.h"

#ifdef __cplusplus
}
#endif

#ifndef MAX_FILENM_LENGTH
#define MAX_FILENM_LENGTH 1024
#endif

namespace yolo{
    image **load_labels(char* lable_dir);

    static float   nms_ = .4;
    static network net_;
    static char**  names_;
    static image** alphabet_;
    static float   thresh_;
    static float   hier_thresh_;

    bool yolo_wrapper::yolo_init_net_cpp(int gpu_id, const char *cfgfile
            , const char *weightfile
            , const char* namelist
            , const char* labeldir
            , float thresh
            , float hier_thresh) {
#ifdef YOLO_GPU
        if (gpu_id >= 0) {
            cudaSetDevice(gpu_id);
        }
#endif
        names_ = get_labels((char*)namelist);

        /// label
        //alphabet_ = load_labels((char*)labeldir);
        thresh_ = thresh;
        hier_thresh_ = hier_thresh;

        /// net
        net_ = parse_network_cfg((char*)cfgfile);
        if (weightfile)
            load_weights(&net_, (char*)weightfile);
        set_batch_network(&net_, 1);

        srand(2222222);

        return true;
    }

    //void* convert_image_jpg(image p)
    //{
    //    image copy = copy_image(p);
    //    if (p.c == 3) rgbgr_image(copy);
    //    int x, y, k;

    //    IplImage *disp = cvCreateImage(cvSize(p.w, p.h), IPL_DEPTH_8U, p.c);
    //    int step = disp->widthStep;
    //    for (y = 0; y < p.h; ++y) {
    //        for (x = 0; x < p.w; ++x) {
    //            for (k = 0; k < p.c; ++k) {
    //                disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy, x, y, k) * 255);
    //            }
    //        }
    //    }
    //    //cvSaveImage(buff, disp, 0);
    //    //cvReleaseImage(&disp);
    //    free_image(copy);

    //    return disp;
    //}

    bool yolo_wrapper::yolo_predict_cpp(void* input, void* output) {
        //printf("Predicted .........................\n");
        int j;
        image im = *(image*)input;
        image sized = letterbox_image(im, net_.w, net_.h);

        layer l = net_.layers[net_.n - 1];

        box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
        for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes + 1, sizeof(float *));

        float *X = sized.data;
        float time = clock();
        network_predict(net_, X);

        get_region_boxes(l, im.w, im.h, net_.w, net_.h, thresh_, probs, boxes, 0, 0, hier_thresh_, 1);
        if (nms_) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms_);

        draw_detections(im, l.w*l.h*l.n, thresh_, boxes, probs, names_, alphabet_, l.classes);

        //save_image(im, "./aaa.jpg");
        //   output = convert_image_jpg;

        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);

        return true;
    }

}
