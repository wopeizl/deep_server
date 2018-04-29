
#include "tf_wrapper.h"
#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/padding_fifo_queue.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using namespace tensorflow;
using namespace std;

namespace tf_wrapper{ 

    Status LoadGraph(const string& graph_file_name, const string& gpu_list,
        std::unique_ptr<tensorflow::Session>* session, tensorflow::GraphDef& graph_def) {
        //  tensorflow::GraphDef graph_def;
        Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
        if (!load_graph_status.ok()) {
            return tensorflow::errors::NotFound("Failed to load compute graph at '",
                graph_file_name, "'");
        }

        tensorflow::SessionOptions options_;
        //tensorflow::GPUOptions g_option;
        //g_option.set_visible_device_list(gpu_list);
        //g_option.set_per_process_gpu_memory_fraction(0.4);
        //g_option.set_allow_growth(true);
        //tensorflow::string bfc("BFC");
        options_.config.set_allow_soft_placement(true);
        //options_.config.mutable_gpu_options()->set_visible_device_list(gpu_list);
        //options_.config.mutable_gpu_options()->set_allow_growth(true);
        //options_.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.4);
        //options_.config.mutable_gpu_options()->set_allocated_allocator_type(&bfc);

        session->reset(tensorflow::NewSession(options_));
        Status session_create_status = (*session)->Create(graph_def);
        if (!session_create_status.ok()) {
            return session_create_status;
        }
        return Status::OK();
    }

    void prepare_tf_detect_result(tensorflow::Tensor& boxes, tensorflow::Tensor& classes, tensorflow::Tensor& scores,
        float scale_factor_w, float scale_factor_h, float threshold,
        std::map<int, std::vector<c_tf_detect_result> > & dst) {
        tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
        tensorflow::TTypes<long long>::Flat classes_flat = classes.flat<long long>();
        TTypes<float, 3>::Tensor obox = boxes.tensor<float, 3>();
        dst.clear();
        for (int i = 0; i < scores_flat.size(); ++i) {
            if (scores_flat(i) >= threshold && scores_flat(i) < 1) {
                c_tf_detect_result nr;
                nr.nclass = classes_flat(i);
                nr.score = scores_flat(i);
                int cx = obox(0, i, 0);
                int cy = obox(0, i, 1);
                int w = obox(0, i, 2);
                int h = obox(0, i, 3);
                int x1 = cx - w / 2;
                int y1 = cy - h / 2;
                int x2 = cx + w / 2;
                int y2 = cy + h / 2;
                x1 /= scale_factor_w;
                x2 /= scale_factor_w;
                y1 /= scale_factor_h;
                y2 /= scale_factor_h;
                nr.r = cv::Rect(x1 < x2 ? x1 : x2, y1 < y2 ? y1 : y2, abs(x1 - x2), abs(y1 - y2));

                auto iter = dst.find(nr.nclass);
                if (iter == dst.end()) {
                    std::vector<c_tf_detect_result> s;
                    s.push_back(nr);
                    dst.insert(std::make_pair(nr.nclass, s));
                }
                else {
                    iter->second.push_back(nr);
                }
            }
        }
    }

    void nms_plus(std::map<int, std::vector<c_tf_detect_result> >& input, std::map<int, std::vector<c_tf_detect_result> >& output, float thresh) {
        output.clear();

        for (auto iter = input.begin(); iter != input.end(); ++iter) {
            std::vector<c_tf_detect_result>& src = iter->second;
            std::sort(src.begin(), src.end(), [](c_tf_detect_result c1, c_tf_detect_result c2) { return c1.score < c2.score; });

            std::vector<c_tf_detect_result> dst;
            while (src.size() > 0)
            {
                // grab the last rectangle
                auto lastElem = --std::end(src);
                const cv::Rect& rect1 = lastElem->r;

                src.erase(lastElem);

                for (auto pos = std::begin(src); pos != std::end(src); )
                {
                    // grab the current rectangle
                    const cv::Rect& rect2 = pos->r;

                    float intArea = (rect1 & rect2).area();
                    float unionArea = rect1.area() + rect2.area() - intArea;
                    float overlap = intArea / unionArea;

                    // if there is sufficient overlap, suppress the current bounding box
                    if (overlap >= thresh)
                    {
                        pos = src.erase(pos);
                    }
                    else
                    {
                        ++pos;
                    }
                }

                dst.push_back(*lastElem);
            }

            output.insert(std::make_pair(iter->first, dst));
        }
    }

    Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
        Tensor* indices, Tensor* scores) {
        auto root = tensorflow::Scope::NewRootScope();
        using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

        string output_name = "top_k";
        TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
        // This runs the GraphDef network definition that we've just constructed, and
        // returns the results in the output tensors.
        tensorflow::GraphDef graph;
        TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

        std::unique_ptr<tensorflow::Session> session(
            tensorflow::NewSession(tensorflow::SessionOptions()));
        TF_RETURN_IF_ERROR(session->Create(graph));
        // The TopK node returns two outputs, the scores and their original indices,
        // so we have to append :0 and :1 to specify them both.
        std::vector<Tensor> out_tensors;
        TF_RETURN_IF_ERROR(session->Run({}, { output_name + ":0", output_name + ":1" },
        {}, &out_tensors));
        *scores = out_tensors[0];
        *indices = out_tensors[1];
        return Status::OK();
    }

    std::unique_ptr<tensorflow::Session> session;
    int32 input_width = 0;
    int32 input_height = 0;
    float thres_hold = 0.6;

    bool tf_wrapper::init(vector<string>& flags) {

        string ckpt = "";
        string graph = "";
        string dev_list = "";

        int argc = flags.size();
        std::vector<char*> cstrings;
        for (size_t i = 0; i < flags.size(); ++i)
            cstrings.push_back(const_cast<char*>(flags[i].c_str()));
        char** argv = &cstrings[0];

        std::vector<Flag> flag_list = {
            Flag("graph", &graph, "graph to be executed"),
            Flag("thres_hold", &thres_hold, "probability thres_hold"),
            Flag("dev_list", &dev_list, "dev_list"),
            Flag("ckpt", &ckpt, "check point"),
            Flag("input_width", &input_width, "resize image to this width in pixels"),
            Flag("input_height", &input_height, "resize image to this height in pixels"),
        };
        string usage = tensorflow::Flags::Usage(argv[0], flag_list);
        const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
        if (!parse_result) {
            LOG(ERROR) << usage;
            return false;
        }

        // We need to call this to set up global state for TensorFlow.
        tensorflow::port::InitMain(argv[0], &argc, &argv);
        if (argc > 1) {
            LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
            return false;
        }

        tensorflow::GraphDef graph_def;

        Status load_graph_status = LoadGraph(graph, dev_list, &session, graph_def);
        if (!load_graph_status.ok()) {
            LOG(ERROR) << load_graph_status;
            return false;
        }

        return false;
    }

    //bool tf_wrapper::init(string model_name, string model_base_path, string model_config_file){
    //    return false;
    //}

    bool tf_wrapper::preprocess() {
        return false;
    }

    bool tf_wrapper::detect_predict(const cv::Mat & input, vector<c_tf_detect_result> & output) {
        if (input.data == nullptr)
            return false;

        cv::Mat img;
        input.convertTo(img, CV_32FC3);

        const int height = img.rows;
        const int width = img.cols;

        float scale_factor_w = (float)input_width / width;
        float scale_factor_h = (float)input_height / height;
        float factor = scale_factor_w < scale_factor_h ? scale_factor_w : scale_factor_h;
        cv::resize(img, img, cv::Size(), factor, factor);

        for (int r = 0; r < img.rows; r++) {
            for (int c = 0; c < img.cols; c++) {
                int offset = (r * img.cols + c) * 3;
                reinterpret_cast<float *>(img.data)[offset + 0] -= 103.939; // B
                reinterpret_cast<float *>(img.data)[offset + 1] -= 116.779;
                reinterpret_cast<float *>(img.data)[offset + 2] -= 123.68;
            }
        }

        cv::Mat baseimg(input_height, input_width, CV_32FC3);
        cv::Rect rect(0, 0, img.cols, img.rows);
        cv::Mat image_roi = baseimg(rect);
        img.copyTo(image_roi);

        Tensor inputImg(tensorflow::DT_FLOAT, { 1,input_height,input_width,3 });
        auto inputImageMapped = inputImg.tensor<float, 4>();
        //Copy all the data over
        for (int y = 0; y < input_height; ++y) {
            const float* source_row = ((float*)baseimg.data) + (y * input_width * 3);
            for (int x = 0; x < input_width; ++x) {
                const float* source_pixel = source_row + (x * 3);
                inputImageMapped(0, y, x, 0) = source_pixel[0];
                inputImageMapped(0, y, x, 1) = source_pixel[1];
                inputImageMapped(0, y, x, 2) = source_pixel[2];
            }
        }
        std::vector<Tensor> outputs;
        std::vector<string> olabels = { "bbox/trimming/bbox","probability/class_idx","probability/score" };

        tensorflow::TensorShape image_input_shape;
        image_input_shape.AddDim(1);
        image_input_shape.AddDim(input_height);
        image_input_shape.AddDim(input_width);
        image_input_shape.AddDim(3);

        tensorflow::TensorShape box_mask_shape;
        box_mask_shape.AddDim(1);
        box_mask_shape.AddDim(16848);
        box_mask_shape.AddDim(1);
        Tensor box_mask_tensor(tensorflow::DataType::DT_FLOAT, box_mask_shape);

        tensorflow::TensorShape box_delta_input_shape;
        box_delta_input_shape.AddDim(1);
        box_delta_input_shape.AddDim(16848);
        box_delta_input_shape.AddDim(4);
        Tensor box_delta_input_tensor(tensorflow::DataType::DT_FLOAT, box_delta_input_shape);

        tensorflow::TensorShape box_input_shape;
        box_input_shape.AddDim(1);
        box_input_shape.AddDim(16848);
        box_input_shape.AddDim(4);
        Tensor box_input_tensor(tensorflow::DataType::DT_FLOAT, box_input_shape);

        tensorflow::TensorShape labels_shape;
        labels_shape.AddDim(1);
        labels_shape.AddDim(16848);
        labels_shape.AddDim(3);
        Tensor labels_tensor(tensorflow::DataType::DT_FLOAT, labels_shape);

        /*
            Status run_status = session->Run({ { "image_input", inputImg },{ "box_mask", box_mask_tensor },{ "box_delta_input", box_delta_input_tensor },{ "box_input", box_input_tensor },{ "labels", labels_tensor } },
                {}, { "fifo_queue_EnqueueMany" }, nullptr);

            if (!run_status.ok()) {
                LOG(ERROR) << "Running model failed: " << run_status;
                return -1;
            }

            run_status = session->Run({},{}, { "batch/fifo_queue_enqueue" }, nullptr);

            if (!run_status.ok()) {
                LOG(ERROR) << "Running model failed: " << run_status;
                return false;
            }

            run_status = session->Run({}, olabels, {}, &outputs);

            if (!run_status.ok()) {
                LOG(ERROR) << "Running model failed: " << run_status;
                return false;
            }
        */
        Status run_status = session->Run({ { "image_input", inputImg } },
            olabels, {}, &outputs);


        if (!run_status.ok()) {
            LOG(ERROR) << "Running model failed: " << run_status;
            return -1;
        }

        Tensor boxes = outputs[0];
        Tensor indices = outputs[1];
        Tensor scores = outputs[2];

        std::map<int, std::vector<c_tf_detect_result> >  src;
        std::map<int, std::vector<c_tf_detect_result> >  dst;
        prepare_tf_detect_result(boxes, indices, scores, factor, factor, thres_hold, src);
        nms_plus(src, dst, 0.4);

        for (auto iter = dst.begin(); iter != dst.end(); ++iter) {
            std::vector<c_tf_detect_result>& src = iter->second;
            std::copy(src.begin(), src.end(), back_inserter(output));
        }

        return true;
    }

    bool tf_wrapper::classify_predict(const cv::Mat & input, vector<c_tf_classify_result> & output) {
        if (input.data == nullptr)
            return false;

        cv::Mat img;
        input.convertTo(img, CV_32FC3);

        const int height = img.rows;
        const int width = img.cols;

        float scale_factor_w = (float)input_width / width;
        float scale_factor_h = (float)input_height / height;
        cv::resize(img, img, cv::Size(), scale_factor_w, scale_factor_h);

        for (int r = 0; r < img.rows; r++) {
            for (int c = 0; c < img.cols; c++) {
                int offset = (r * img.cols + c) * 3;
                reinterpret_cast<float *>(img.data)[offset + 0] /= 255.0; // B
                reinterpret_cast<float *>(img.data)[offset + 1] /= 255.0;
                reinterpret_cast<float *>(img.data)[offset + 2] /= 255.0;
            }
        }

        Tensor inputImg(tensorflow::DT_FLOAT, { 1,input_height,input_width,3 });
        auto inputImageMapped = inputImg.tensor<float, 4>();
        //Copy all the data over
        for (int y = 0; y < input_height; ++y) {
            const float* source_row = ((float*)img.data) + (y * input_width * 3);
            for (int x = 0; x < input_width; ++x) {
                const float* source_pixel = source_row + (x * 3);
                inputImageMapped(0, y, x, 0) = source_pixel[0];
                inputImageMapped(0, y, x, 1) = source_pixel[1];
                inputImageMapped(0, y, x, 2) = source_pixel[2];
            }
        }

        // Actually run the image through the model.
        std::vector<Tensor> outputs;
        string input_layer = "input";
        string output_layer = "InceptionV4/Logits/Predictions";

        Status run_status = session->Run({ { input_layer, inputImg } },
        { output_layer }, {}, &outputs);
        if (!run_status.ok())
        {
            LOG(ERROR) << "Running model failed: " << run_status;
            return false;
        }

        tensorflow::TTypes<float>::Flat o_scores_flat = outputs[0].flat<float>();
        const int how_many_labels = std::min(5, static_cast<int>(o_scores_flat.size()));
        Tensor indices;
        Tensor scores;
        GetTopLabels(outputs, how_many_labels, &indices, &scores);
        tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
        tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
        for (int pos = 0; pos < how_many_labels; ++pos) {
            c_tf_classify_result r;
            r.score = scores_flat(pos);
            r.nclass = indices_flat(pos);
            output.push_back(r);
        }

        return true;
    }

    bool tf_wrapper::postprocess() {
        return false;
    }
}

