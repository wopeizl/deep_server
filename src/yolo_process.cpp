
#include "yolo_process.hpp"

bool yolo_process::init(int gpu_id, const char *cfgfile
    , const char *weightfile
    , const char* namelist
    , const char* labeldir
    , float thresh
    , float hier_thresh) {

    if (!boost::filesystem::exists(cfgfile)) {
        DEEP_LOG_ERROR("File : " + string(cfgfile) + " Not existed ! ");
        return false;
    }
    if (!boost::filesystem::exists(weightfile)) {
        DEEP_LOG_ERROR("File : " + string(weightfile) + " Not existed ! ");
        return false;
    }
    if (!boost::filesystem::exists(namelist)) {
        DEEP_LOG_ERROR("File : " + string(namelist) + " Not existed ! ");
        return false;
    }

#ifndef SELF_YOLO_WRAPPER
    yolo_init_net(gpu_id, cfgfile, weightfile, namelist, labeldir, thresh, hier_thresh);
#else
    wrapper->yolo_init_net_cpp(gpu_id, cfgfile, weightfile, namelist, labeldir, thresh, hier_thresh);
#endif

    return true;
}

bool yolo_process::prepare(const cv::Mat &img, image* output) {
    unsigned char *data = (unsigned char *)img.data;
    int h = img.rows;
    int w = img.cols;
    int c = img.channels();
    int step = img.step;
    image im = make_image(w, h, c);
    int i, j, k, count = 0;;

    for (k = 0; k < c; ++k) {
        for (i = 0; i < h; ++i) {
            for (j = 0; j < w; ++j) {
                im.data[count++] = data[i*step + j*c + k] / 255.;
            }
        }
    }

    rgbgr_image(im);

    *output = im;

    return true;
}

bool yolo_process::postprocess(image im, cv::Mat &output) {

    int i, y, x, k;
    for (i = 0; i < im.w*im.h*im.c; ++i) {
        if (im.data[i] < 0) im.data[i] = 0;
        if (im.data[i] > 1) im.data[i] = 1;
    }
    for (i = 0; i < im.w*im.h; ++i) {
        float swap = im.data[i];
        im.data[i] = im.data[i + im.w*im.h * 2];
        im.data[i + im.w*im.h * 2] = swap;
    }

    output = cv::Mat(im.h, im.w, CV_8UC3);

    int step = output.step;
    for (y = 0; y < im.h; ++y) {
        for (x = 0; x < im.w; ++x) {
            for (k = 0; k < im.c; ++k) {
                output.data[y*step + x*im.c + k] = (unsigned char)(im.data[k*im.h*im.w + y*im.w + x] * 255);
            }
        }
    }

    return true;
}

bool yolo_process::predict(image* input, cv::Mat& output) {
    image out;

#ifndef SELF_YOLO_WRAPPER
    yolo_predict(input, &out);
#else
    wrapper->yolo_predict_cpp(input, &out);
#endif

    return true;
}

namespace deep_server {

    yolo_processor::yolo_processor(actor_config& cfg,
        std::string  n, actor s, connection_handle handle, actor cm)
        : event_based_actor(cfg),
        name_(std::move(n)), bk(s), handle_(handle), cf_manager(cm) {
        deepconfig& dcfg = (deepconfig&)cfg.host->system().config();
        http_mode = dcfg.http_mode;

        pyolo_data.reset(new yolo_data());
        pyolo_data->time_consumed.whole_time = 0.f;
        pyolo_data->self = this;
        //set_default_handler(skip);

        prepare_.assign(
            [=](prepare_atom, uint64 datapointer) {

            std::chrono::time_point<clock_> beg_ = clock_::now();

            if (!yolo_process::prepare(pyolo_data->cv_image, &pyolo_data->input)) {
                fault("Fail to prepare image£¡");
                return;
            }

            double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
            pyolo_data->time_consumed.prepare_time = elapsed;
            pyolo_data->time_consumed.whole_time += elapsed;
            DEEP_LOG_INFO("prepare caffe consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

            become(preprocess_);
            send(this, preprocess_atom::value, (uint64)(uint64*)pyolo_data.get());
        }
        );

        preprocess_.assign(
            [=](preprocess_atom, uint64 datapointer) {

            become(processor_);
            send(cf_manager, deep_manager_process_atom::value, this, datapointer);
        }
        );

        processor_.assign(
            [=](process_atom, uint64 datapointer) {

            DEEP_LOG_INFO(name_ + " starts to process_atom.");

            yolo_data* data = (yolo_data*)pyolo_data.get();

            connection_handle handle = data->handle;

            become(downloader_);
            vector<unsigned char> odata;
            send(this, downloader_atom::value, handle);
        }
        );

        downloader_.assign(
            [=](downloader_atom, connection_handle handle) {

            DEEP_LOG_INFO(name_ + " starts to downloader_atom.");

            if (pyolo_data->method == input_m::YOLO) {
                if (http_mode) {
                    if (!cvprocess::writeImage(pyolo_data->output, pyolo_data->h_out.bdata)) {
                        fault("fail to cvprocess::process_caffe_result ! ");
                        return;
                    }
                }
                else {
                    if (!cvprocess::writeImage(pyolo_data->output, pyolo_data->t_out.bdata)) {
                        fault("fail to cvprocess::process_caffe_result ! ");
                        return;
                    }
                }
            }

            // yolo will output the processed png all the time
            pyolo_data->resDataType = deep_server::CV_POST_PNG;

            if (http_mode) {
                pyolo_data->h_out.ts = pyolo_data->time_consumed;
                pyolo_data->h_out.sdata = base64_encode(pyolo_data->h_out.bdata.data(), pyolo_data->h_out.bdata.size());
                send(bk, handle, output_atom::value, pyolo_data->resDataType, pyolo_data->callback, pyolo_data->h_out);
            }
            else {
                pyolo_data->t_out.ts = pyolo_data->time_consumed;
                send(bk, handle, output_atom::value, pyolo_data->resDataType, pyolo_data->callback, pyolo_data->t_out);
            }
        }
        );
    }

    behavior yolo_processor::make_behavior() {
        if (http_mode) {
            return (
                [=](input_atom, std::string& data) {
                DEEP_LOG_INFO(name_ + " starts to input_atom.");
                //become(uploader_);
                //send(this, uploader_atom::value);

                Json::Reader reader;
                Json::Value value;

                std::chrono::time_point<clock_> beg_ = clock_::now();
                if (!reader.parse(data, value) || value.isNull() || !value.isObject()) {
                    fault("invalid json");
                    return;
                }
                double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                pyolo_data->time_consumed.jsonparse_time = elapsed;
                pyolo_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("parse json consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                std::string out = value.get("data", "").asString();
                vector<unsigned char> idata;
                beg_ = clock_::now();
                idata = base64_decode(out);
                if (out.length() == 0 || idata.size() == 0) {
                    fault("invalid json property : data.");
                    return;
                }
                elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                pyolo_data->time_consumed.decode_time = elapsed;
                pyolo_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("base64 decode image consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                pyolo_data->callback = value.get("callback", "").asString();
                pyolo_data->resDataType = value.get("res_dataT", "").asInt();

                if (pyolo_data->resDataType < 0 || pyolo_data->resDataType > INVALID_OUTDATA) {
                    fault("invalid response data type!!");
                    return;
                }

                std::string method = value.get("method", "").asString();
                if (method == "flip") {
                    if (!cvprocess::flip(idata, pyolo_data->h_out.bdata)) {
                        fault("Fail to process pic by " + method + ", please check£¡");
                        return;
                    }

                    pyolo_data->method = input_m::CV_FLIP;

                    become(downloader_);
                    send(this, downloader_atom::value, handle_);
                }
                else if (method == "yolo") {

                    std::chrono::time_point<clock_> beg_ = clock_::now();
                    if (!cvprocess::readImage(idata, pyolo_data->cv_image)) {
                        fault("Fail to read image£¡");
                        return;
                    }

                    double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                    pyolo_data->time_consumed.cvreadimage_time = elapsed;
                    pyolo_data->time_consumed.whole_time += elapsed;
                    DEEP_LOG_INFO("cv read image consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                    pyolo_data->handle = handle_;

                    if (!verifyCV_Image(pyolo_data->cv_image)) {
                        fault("invalid image format £¡");
                        return;
                    }

                    pyolo_data->method = input_m::YOLO;

                    become(prepare_);
                    send(this, prepare_atom::value, (uint64)(uint64*)pyolo_data.get());
                }
                else {
                    fault("invalid method : £¡" + method);
                    return;
                }
            });
        }
        else {
            return ([=](input_atom, int dataType, int resDataType, vector<unsigned char> idata) {
                pyolo_data->handle = handle_;
                pyolo_data->resDataType = resDataType;

                std::chrono::time_point<clock_> beg_ = clock_::now();

                if (dataType == deep_server::dataType::PNG) {
                    if (!cvprocess::readImage(idata, pyolo_data->cv_image)) {
                        fault("Fail to read image£¡");
                        return;
                    }
                }
                else if (dataType == deep_server::dataType::CV_IMAGE) {
                    pyolo_data->cv_image = *(cv::Mat*)idata.data();
                }
                else {
                    fault("invalid data type : £¡" + dataType);
                    return;
                }

                if (!verifyCV_Image(pyolo_data->cv_image)) {
                    fault("invalid image format £¡");
                    return;
                }

                double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                pyolo_data->time_consumed.cvreadimage_time = elapsed;
                pyolo_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("cv read image consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                become(prepare_);
                send(this, prepare_atom::value, (uint64)(uint64*)pyolo_data.get());
            }
            );

        }
    }
}