
#include "caffe_process.hpp"

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on the given device ID, Empty is CPU");
DEFINE_string(cpu, "",
    "Optional; run in CPU mode");
DEFINE_string(solver_count, "",
    "count for caffe solver count");
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

bool caffe_process::initialized = false;
atomic_flag caffe_flag = ATOMIC_FLAG_INIT;

bool caffe_process::init(vector<string>& flags, int gpu_id) {

    int argc = flags.size();
    std::vector<char*> cstrings;
    for (size_t i = 0; i < flags.size(); ++i)
        cstrings.push_back(const_cast<char*>(flags[i].c_str()));
    char** argv = &cstrings[0];

    if (!initialized) {
        while (caffe_flag.test_and_set());
        if (!initialized) {
            FLAGS_alsologtostderr = 1;
            gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
            caffe::GlobalInit(&argc, &argv);
            initialized = true;
        }
        caffe_flag.clear();
    }

    //CHECK(FLAGS_gpu.size() == 0 || FLAGS_gpu.size() == 1) << "Can only support one gpu or none";

    //int gpu_id = -1;
    //if (FLAGS_gpu.size() > 0)
    //    gpu_id = boost::lexical_cast<int>(FLAGS_gpu);

    int solver_count = 1;
    if (FLAGS_solver_count.size() > 0)
        solver_count = boost::lexical_cast<int>(FLAGS_solver_count);
    solver_count = solver_count > 0 ? solver_count : 1;

    if (gpu_id >= 0) {
#ifndef CPU_ONLY
        caffe::Caffe::SetDevice(gpu_id);
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
        Caffe::set_solver_count(solver_count);
        Caffe::set_solver_rank(gpu_id);
        Caffe::set_multiprocess(true);
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

    wrapper = new FRCNN_API::Frcnn_wrapper(proto_file, model_file, config_file);

    return true;
}

bool caffe_process::prepare(const cv::Mat &input, cv::Mat &img) {
    return wrapper->prepare(input, img);
}

bool caffe_process::preprocess(const cv::Mat &img, vector<boost::shared_ptr<Blob<float> >>& input) {
    return wrapper->preprocess(img, input);
}

bool caffe_process::predict(vector<boost::shared_ptr<Blob<float> >>& input, vector<boost::shared_ptr<Blob<float> >>& output) {
    return wrapper->predict(input, output);
}

bool caffe_process::postprocess(const cv::Mat &img, vector<boost::shared_ptr<Blob<float> >>& input, std::vector<caffe::Frcnn::BBox<float> > &output) {
    return wrapper->postprocess(img, input, output);
}


