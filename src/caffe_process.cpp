
#include "caffe_process.hpp"

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on the given device ID, Empty is CPU");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "Trained Model By Faster RCNN End-to-End Pipeline.");
DEFINE_string(default_c, "",
    "Default config file path.");
DEFINE_string(image_list, "",
    "Optional;Test images list.");
DEFINE_string(image_root, "",
    "Optional;Test images root directory.");
DEFINE_string(out_file, "",
    "Optional;Output images file.");
DEFINE_int32(max_per_image, 100,
    "Limit to max_per_image detections *over all classes*");

bool caffe_process::init(vector<string>& flags) {
    FLAGS_alsologtostderr = 1;
    gflags::SetVersionString(AS_STRING(CAFFE_VERSION));

    int argc = flags.size();
    std::vector<char*> cstrings;
    for (size_t i = 0; i < flags.size(); ++i)
        cstrings.push_back(const_cast<char*>(flags[i].c_str()));
    char** argv = &cstrings[0];

    caffe::GlobalInit(&argc, &argv);
    CHECK(FLAGS_gpu.size() == 0 || FLAGS_gpu.size() == 1) << "Can only support one gpu or none";
    int gpu_id = -1;
    if (FLAGS_gpu.size() > 0)
        gpu_id = boost::lexical_cast<int>(FLAGS_gpu);

    if (gpu_id >= 0) {
#ifndef CPU_ONLY
        caffe::Caffe::SetDevice(gpu_id);
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
#else
        LOG(FATAL) << "CPU ONLY MODEL, BUT PROVIDE GPU ID";
#endif
    }
    else {
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
    }

    std::string proto_file = FLAGS_model.c_str();
    std::string model_file = FLAGS_weights.c_str();
    std::string config_file = FLAGS_default_c.c_str();
    const int max_per_image = FLAGS_max_per_image;

    detector = new FRCNN_API::Detector(proto_file, model_file, config_file);

    return true;
}

bool caffe_process::predict(cv::Mat& cv_image, std::vector<caffe::Frcnn::BBox<float>>& results) {
    detector->predict(cv_image, results);
    return true;
}
