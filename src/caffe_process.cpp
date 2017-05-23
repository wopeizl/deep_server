
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

    if (!boost::filesystem::exists(proto_file)) {
        DEEP_LOG_ERROR("File : " + proto_file + " Not existed ! ");
        return false;
    }
    if (!boost::filesystem::exists(model_file)) {
        DEEP_LOG_ERROR("File : " + model_file + " Not existed ! ");
        return false;
    }
    if (!boost::filesystem::exists(config_file)) {
        DEEP_LOG_ERROR("File : " + config_file + " Not existed ! ");
        return false;
    }

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

//bool caffe_process::predict(vector<boost::shared_ptr<Blob<float> >>& input, vector<boost::shared_ptr<Blob<float> >>& output) {
//    return wrapper->predict(input, output);
//}

bool caffe_process::postprocess(const cv::Mat &img, vector<boost::shared_ptr<Blob<float> >>& input, std::vector<caffe::Frcnn::BBox<float> > &output) {
    return wrapper->postprocess(img, input, output);
}

bool caffe_process::predict_b(const std::vector<cv::Mat >& img_in, vector<boost::shared_ptr<Blob<float> > >& output) {
    return wrapper->predict_b(img_in, output);
}

bool caffe_process::postprocess_b(const std::vector<cv::Mat >& img_in, int imat, vector<boost::shared_ptr<Blob<float> > >& input, std::vector<caffe::Frcnn::BBox<float> > &results) {
    return wrapper->postprocess_b(img_in, imat, input, results);
}

namespace deep_server {
    caffe_processor::caffe_processor(actor_config& cfg,
        std::string  n, actor s, connection_handle handle, actor cm)
        : event_based_actor(cfg),
        name_(std::move(n)), bk(s), handle_(handle), cf_manager(cm) {

        deepconfig& dcfg = (deepconfig&)cfg.host->system().config();
        http_mode = dcfg.http_mode;

        this->set_down_handler([=](const down_msg& dm) {
            this->quit();
        });

        pcaffe_data.reset(new caffe_data());
        pcaffe_data->time_consumed.whole_time = 0.f;
        pcaffe_data->self = this;
        //set_default_handler(skip);

        prepare_.assign(
            [=](prepare_atom, uint64 datapointer) {

            std::chrono::time_point<clock_> beg_ = clock_::now();
            //if (!FRCNN_API::Frcnn_wrapper::prepare(pcaffe_data->cv_image, pcaffe_data->out_image)) {
            //    fault("Fail to prepare image£¡");
            //    return;
            //}
            double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
            pcaffe_data->time_consumed.prepare_time = elapsed;
            pcaffe_data->time_consumed.whole_time += elapsed;
            DEEP_LOG_INFO("prepare caffe consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

            become(preprocess_);
            send(this, preprocess_atom::value, datapointer);
        }
        );

        preprocess_.assign(
            [=](preprocess_atom, uint64 datapointer) {

            //std::chrono::time_point<clock_> beg_ = clock_::now();
            //if (!FRCNN_API::Frcnn_wrapper::preprocess(pcaffe_data->out_image, pcaffe_data->input)) {
            //    fault("Fail to preprocess image£¡");
            //    return;
            //}
            //double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
            //pcaffe_data->time_consumed.preprocess_time = elapsed;
            //pcaffe_data->time_consumed.whole_time += elapsed;
            //DEEP_LOG_INFO("preprocess caffe consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

            become(processor_);
            send(cf_manager, deep_manager_process_atom::value, this, datapointer);
        }
        );

        processor_.assign(
            [=](process_atom, uint64 datapointer) {

            DEEP_LOG_INFO(name_ + " starts to process_atom.");

            caffe_data* data = (caffe_data*)pcaffe_data.get();

            std::chrono::time_point<clock_> beg_ = clock_::now();

            if (pcaffe_data->resDataType == deep_server::CV_POST_IMAGE
                || pcaffe_data->resDataType == deep_server::CV_POST_PNG) {
                //if (!FRCNN_API::Frcnn_wrapper::postprocess(data->cv_image, data->output, data->results)) {
                //    fault("fail to Frcnn_wrapper::postprocess ! ");
                //    return;
                //}
                double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                pcaffe_data->time_consumed.postprocess_time = elapsed;
                pcaffe_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("postprocess caffe consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                beg_ = clock_::now();
                if (!cvprocess::process_caffe_result(data->results, data->cv_image)) {
                    fault("fail to cvprocess::process_caffe_result ! ");
                    return;
                }
                elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                pcaffe_data->time_consumed.postprocess_time += elapsed;
                pcaffe_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("process caffe result consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                beg_ = clock_::now();
                if (http_mode) {
                    if (!cvprocess::writeImage(data->cv_image, pcaffe_data->h_out.bdata)) {
                        fault("fail to cvprocess::process_caffe_result ! ");
                        return;
                    }
                }
                else if (!http_mode && pcaffe_data->resDataType == deep_server::CV_POST_PNG) {
                    pcaffe_data->t_out.bdata.resize(*data->cv_image.size);
                    std::memcpy(pcaffe_data->t_out.bdata.data(), data->cv_image.data, *data->cv_image.size);
                }
                else if (!http_mode && pcaffe_data->resDataType == deep_server::CV_IMAGE) {
                    pcaffe_data->t_out.bdata.resize(*data->cv_image.size);
                    std::memcpy(pcaffe_data->t_out.bdata.data(), data->cv_image.data, *data->cv_image.size);
                }
                elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                pcaffe_data->time_consumed.writeresult_time = elapsed;
                pcaffe_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("write result consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");
            }
            else if (pcaffe_data->resDataType == deep_server::FRCNN_RESULT) {
                beg_ = clock_::now();

                for (int ir = 0; ir < data->results.size(); ir++) {
                    if (data->results[ir].confidence > 0.9){
                        Caffe_result r;
                        r.set_x(data->results[ir][0]);
                        r.set_y(data->results[ir][1]);
                        r.set_w(data->results[ir][2]);
                        r.set_h(data->results[ir][3]);
                        r.set_class_(7);
                        r.set_score(data->results[ir].confidence);

                        if (http_mode) {
                            pcaffe_data->h_out.caffe_result.push_back(r);
                        }
                        else{
                            pcaffe_data->t_out.caffe_result.push_back(r);
                        }
                    }
                }

                double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                pcaffe_data->time_consumed.writeresult_time = elapsed;
                pcaffe_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("write result consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");
            }
            else {
                fault("invalid return type!!!");
                return;
            }

            connection_handle handle = data->handle;

            become(downloader_);
            send(this, downloader_atom::value, handle);
        }
        );

        downloader_.assign(
            [=](downloader_atom, connection_handle handle) {

            DEEP_LOG_INFO(name_ + " starts to downloader_atom.");

            if (http_mode) {
                pcaffe_data->h_out.ts = pcaffe_data->time_consumed;
                if(pcaffe_data->resDataType == deep_server::FRCNN_RESULT){
                }
                else {
                    pcaffe_data->h_out.sdata = base64_encode(pcaffe_data->h_out.bdata.data(), pcaffe_data->h_out.bdata.size());
                }
                send(bk, handle, output_atom::value, pcaffe_data->resDataType, pcaffe_data->callback, pcaffe_data->h_out);
            }
            else {
                pcaffe_data->h_out.ts = pcaffe_data->time_consumed;
                send(bk, output_atom::value, pcaffe_data->resDataType, pcaffe_data->callback, pcaffe_data->t_out);
            }
        }
        );
    }

    behavior caffe_processor::make_behavior() {
        if (http_mode) {
            return (
                [=](input_atom, std::string& data) {
                DEEP_LOG_INFO(name_ + " starts to input_atom.");

                Json::Reader reader;
                Json::Value value;

                std::chrono::time_point<clock_> beg_ = clock_::now();
                if (!reader.parse(data, value) || value.isNull() || !value.isObject()) {
                    fault("invalid json");
                    return;
                }
                double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                pcaffe_data->time_consumed.jsonparse_time = elapsed;
                pcaffe_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("parse json consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                pcaffe_data->callback = value.get("callback", "").asString();
                if (!value.get("res_dataT", "").isInt()) {
                    fault("invalid response data type!!");
                     return;
                }
                pcaffe_data->resDataType = value.get("res_dataT", "").asInt();
                if (pcaffe_data->resDataType < 0 || pcaffe_data->resDataType > INVALID_OUTDATA) {
                    fault("invalid response data type!!");
                     return;
                }

                std::string out = value.get("data", "").asString();
                vector<unsigned char> idata;
                beg_ = clock_::now();
                idata = base64_decode(out);
                if (out.length() == 0 || idata.size() == 0) {
                    fault("invalid json property : data.");
                    return;
                }
                elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                pcaffe_data->time_consumed.decode_time = elapsed;
                pcaffe_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("base64 decode image consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                std::string method = value.get("method", "").asString();
                if (method == "flip") {
                    if (!cvprocess::flip(idata, pcaffe_data->h_out.bdata)) {
                        fault("Fail to process pic by " + method + ", please check£¡");
                        return;
                    }

                    pcaffe_data->h_out.sdata = base64_encode(pcaffe_data->h_out.bdata.data(), pcaffe_data->h_out.bdata.size());

                    become(downloader_);
                    send(this, downloader_atom::value, handle_);
                }
                else if (method == "caffe") {

                    std::chrono::time_point<clock_> beg_ = clock_::now();
                    if (!cvprocess::readImage(idata, pcaffe_data->cv_image)) {
                        fault("Fail to read image£¡");
                        return;
                    }
                    double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                    pcaffe_data->time_consumed.cvreadimage_time = elapsed;
                    pcaffe_data->time_consumed.whole_time += elapsed;
                    DEEP_LOG_INFO("cv read image consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                    pcaffe_data->handle = handle_;

                    if (!verifyCV_Image(pcaffe_data->cv_image)) {
                        fault("invalid image format £¡");
                        return;
                    }

                    pcaffe_data->h_out.info.set_width(pcaffe_data->cv_image.cols);
                    pcaffe_data->h_out.info.set_height(pcaffe_data->cv_image.rows);
                    pcaffe_data->h_out.info.set_channel(pcaffe_data->cv_image.channels());

                    become(prepare_);
                    send(this, prepare_atom::value, (uint64)(uint64*)pcaffe_data.get());
                }
                else {
                    fault("invalid method : £¡" + method);
                    return;
                }
            });
        }
        else {
            return ([=](input_atom, int dataType, int resDataType, vector<unsigned char> idata) {
                pcaffe_data->handle = handle_;
                pcaffe_data->resDataType = resDataType;

                if (dataType == deep_server::dataType::CV_IMAGE) {
                    pcaffe_data->cv_image = *(cv::Mat*)idata.data();
                }
                else if (dataType == deep_server::dataType::PNG) {
                    if (!cvprocess::readImage(idata, pcaffe_data->cv_image)) {
                        fault("Fail to read image£¡");
                        return;
                    }
                }
                else {
                    fault("invalid data type : £¡" + dataType);
                    return;
                }

                if (!verifyCV_Image(pcaffe_data->cv_image)) {
                    fault("invalid image format £¡");
                    return;
                }

                pcaffe_data->t_out.info.set_width(pcaffe_data->cv_image.cols);
                pcaffe_data->t_out.info.set_height(pcaffe_data->cv_image.rows);
                pcaffe_data->t_out.info.set_channel(pcaffe_data->cv_image.channels());

                become(prepare_);
                send(this, prepare_atom::value, (uint64)(uint64*)pcaffe_data.get());
            }
            );
        }
    }
}