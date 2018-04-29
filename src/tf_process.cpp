
#include "tf_process.hpp"

bool tf_process::init(vector<string>& flags, int gpu_id) {

    wrapper.init(flags);
    
    return true;
}

//bool tf_process::init(string model_name, string model_base_path, string model_config_file) {
//
//    wrapper.init(model_name, model_base_path, model_config_file);
//    
//    return true;
//}

bool tf_process::prepare() {
    return false;
}

bool tf_process::preprocess() {
    return false;
}

bool tf_process::detect_predict(cv::Mat & img, vector<c_tf_detect_result>& output) {
    return wrapper.detect_predict(img, output);
}

bool tf_process::classify_predict(cv::Mat & img, vector<c_tf_classify_result>& output) {
    return wrapper.classify_predict(img, output);
}

bool tf_process::postprocess(const std::vector<vector<unsigned char> >& img_in, int imat,
    vector<c_tf_detect_result>& output, vector<deep_server::Tf_detect_result>& results) {
    for (auto iter = output.begin(); iter != output.end(); ++iter) {
        deep_server::Tf_detect_result r;
        r.set_class_(iter->nclass);
        r.set_score(iter->score);
        r.set_x(iter->r.x);
        r.set_y(iter->r.y);
        r.set_w(iter->r.x + iter->r.width);
        r.set_h(iter->r.y + iter->r.height);
        results.push_back(r);
    }

    return true;
}

bool tf_process::postprocess(const std::vector<vector<unsigned char> >& img_in, int imat,
    vector<c_tf_classify_result>& output, vector<deep_server::Tf_classify_result>& results) {
    for (auto iter = output.begin(); iter != output.end(); ++iter) {
        deep_server::Tf_classify_result r;
        r.set_class_(iter->nclass);
        r.set_score(iter->score);
        results.push_back(r);
    }

    return true;
}

namespace deep_server {
    tf_processor::tf_processor(actor_config& cfg,
        std::string  n, actor s, connection_handle handle, actor cm)
        : event_based_actor(cfg),
        name_(std::move(n)), bk(s), handle_(handle), cf_manager(cm) {

        deepconfig& dcfg = (deepconfig&)cfg.host->system().config();
        http_mode = dcfg.http_mode;

        this->set_down_handler([=](const down_msg& dm) {
            this->quit();
        });

        ptf_data.reset(new tf_data());
        ptf_data->time_consumed.whole_time = 0.f;
        ptf_data->self = this;

        prepare_.assign(
            [=](prepare_atom, uint64 datapointer) {

            std::chrono::time_point<clock_> beg_ = clock_::now();

            double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
            ptf_data->time_consumed.prepare_time = elapsed;
            ptf_data->time_consumed.whole_time += elapsed;
            DEEP_LOG_INFO("prepare tensorflow consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

            become(preprocess_);
            send(this, preprocess_atom::value, datapointer);
        }
        );

        preprocess_.assign(
            [=](preprocess_atom, uint64 datapointer) {

            //std::chrono::time_point<clock_> beg_ = clock_::now();

            //double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
            //ptf_data->time_consumed.preprocess_time = elapsed;
            //ptf_data->time_consumed.whole_time += elapsed;
            //DEEP_LOG_INFO("preprocess tensorflow consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

            become(processor_);
            send(cf_manager, deep_manager_process_atom::value, backend_mode_t::tensorflow, this, datapointer);
        }
        );

        processor_.assign(
            [=](process_atom, uint64 datapointer) {

            DEEP_LOG_INFO(name_ + " starts to process_atom.");

            tf_data* data = (tf_data*)ptf_data.get();

            std::chrono::time_point<clock_> beg_ = clock_::now();

            if (ptf_data->resDataType == deep_server::CV_POST_IMAGE
                || ptf_data->resDataType == deep_server::CV_POST_PNG) {
                double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                ptf_data->time_consumed.postprocess_time = elapsed;
                ptf_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("postprocess tensorflow consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                beg_ = clock_::now();

                if (!cvprocess::process_tf_detect_result(data->detect_results, data->cv_image)) {
                    fault("fail to cvprocess::process_caffe_result ! ");
                    return;
                }

                elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                ptf_data->time_consumed.postprocess_time += elapsed;
                ptf_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("process tensorflow result consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                beg_ = clock_::now();
                if (http_mode) {
                    if (!cvprocess::writeImage(data->cv_image, ptf_data->h_out.bdata)) {
                        fault("fail to cvprocess::process_tensorflow_result ! ");
                        return;
                    }
                }
                else if (!http_mode && ptf_data->resDataType == deep_server::CV_POST_PNG) {
                    if (!cvprocess::writeImage(ptf_data->cv_image, ptf_data->t_out.bdata)) {
                        fault("tensorflow fail to cvprocess::writeImage ! ");
                        return;
                    }
                }
                else if (!http_mode && ptf_data->resDataType == deep_server::CV_POST_IMAGE) {
                    ptf_data->t_out.bdata.resize(*data->cv_image.size);
                    std::memcpy(ptf_data->t_out.bdata.data(), data->cv_image.data, *data->cv_image.size);
                }
                elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                ptf_data->time_consumed.writeresult_time = elapsed;
                ptf_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("write result consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");
            }
            else if (ptf_data->resDataType == deep_server::TF_DETECT_RESULT) {
                beg_ = clock_::now();

                for (int ir = 0; ir < data->detect_results.size(); ir++) {
                    Tf_detect_result r;
                    r.set_x(data->detect_results[ir].x());
                    r.set_y(data->detect_results[ir].y());
                    r.set_w(data->detect_results[ir].w());
                    r.set_h(data->detect_results[ir].h());
                    r.set_score(data->detect_results[ir].score());
                    r.set_class_(data->detect_results[ir].class_());

                    if (http_mode) {
                        ptf_data->h_out.tf_detect_result.push_back(r);
                    }
                    else {
                        ptf_data->t_out.tf_detect_result.push_back(r);
                    }
                }

                double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                ptf_data->time_consumed.writeresult_time = elapsed;
                ptf_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("write result consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");
            }
            else if (ptf_data->resDataType == deep_server::TF_CLASSIFY_RESULT) {
                beg_ = clock_::now();

                for (int ir = 0; ir < data->classify_results.size(); ir++) {
                    Tf_classify_result r;
                    r.set_score(data->classify_results[ir].score());
                    r.set_class_(data->classify_results[ir].class_());

                    if (http_mode) {
                        ptf_data->h_out.tf_classify_result.push_back(r);
                    }
                    else {
                        ptf_data->t_out.tf_classify_result.push_back(r);
                    }
                }

                double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                ptf_data->time_consumed.writeresult_time = elapsed;
                ptf_data->time_consumed.whole_time += elapsed;
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
                ptf_data->h_out.ts = ptf_data->time_consumed;
                if(ptf_data->resDataType == deep_server::FRCNN_RESULT){
                }
                else {
                    ptf_data->h_out.sdata = base64_encode(ptf_data->h_out.bdata.data(), ptf_data->h_out.bdata.size());
                }
                send(bk, handle, output_atom::value, ptf_data->resDataType, ptf_data->callback, ptf_data->h_out);
            }
            else {
                ptf_data->t_out.ts = ptf_data->time_consumed;
                send(bk, handle, output_atom::value, ptf_data->resDataType, ptf_data->callback, ptf_data->t_out);
            }

            release();
        }
        );
    }

    behavior tf_processor::make_behavior() {
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
                ptf_data->time_consumed.jsonparse_time = elapsed;
                ptf_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("parse json consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                ptf_data->callback = value.get("callback", "").asString();
                if (!value.get("res_dataT", "").isInt()) {
                    fault("invalid response data type!!");
                     return;
                }
                ptf_data->resDataType = value.get("res_dataT", "").asInt();
                if (ptf_data->resDataType < 0 || ptf_data->resDataType > INVALID_OUTDATA) {
                    fault("invalid response data type!!");
                     return;
                }

                std::string out = value.get("data", "").asString();

                beg_ = clock_::now();
                ptf_data->input = base64_decode(out);
                if (out.length() == 0 || ptf_data->input.size() == 0) {
                    fault("invalid json property : data.");
                    return;
                }
                elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                ptf_data->time_consumed.decode_time = elapsed;
                ptf_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("base64 decode image consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                std::string method = value.get("method", "").asString();
                if (method == "flip") {
                    if (!cvprocess::flip(ptf_data->input, ptf_data->h_out.bdata)) {
                        fault("Fail to process pic by " + method + ", please check£¡");
                        return;
                    }

                    ptf_data->h_out.sdata = base64_encode(ptf_data->h_out.bdata.data(), ptf_data->h_out.bdata.size());

                    become(downloader_);
                    send(this, downloader_atom::value, handle_);
                }
                else if (method == "tensorflow") {

                    std::chrono::time_point<clock_> beg_ = clock_::now();
                    if (!cvprocess::readImage(ptf_data->input, ptf_data->cv_image)) {
                        fault("tensorflow Fail to read image£¡");
                        return;
                    }

                    double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                    ptf_data->time_consumed.cvreadimage_time = elapsed;
                    ptf_data->time_consumed.whole_time += elapsed;
                    DEEP_LOG_INFO("cv read image consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                    ptf_data->handle = handle_;

                    if (!verifyCV_Image(ptf_data->cv_image)) {
                        fault("invalid image format £¡");
                        return;
                    }

                    ptf_data->h_out.info.set_width(ptf_data->cv_image.cols);
                    ptf_data->h_out.info.set_height(ptf_data->cv_image.rows);
                    ptf_data->h_out.info.set_channel(ptf_data->cv_image.channels());

                    become(prepare_);
                    send(this, prepare_atom::value, (uint64)(uint64*)ptf_data.get());
                }
                else {
                    fault("invalid method : £¡" + method);
                    return;
                }
            });
        }
        else {
            return ([=](input_atom, int dataType, int resDataType, vector<unsigned char> idata) {
                ptf_data->handle = handle_;
                ptf_data->resDataType = resDataType;
                std::chrono::time_point<clock_> beg_ = clock_::now();

                if (dataType == deep_server::dataType::CV_IMAGE) {
                    ptf_data->cv_image = *(cv::Mat*)idata.data();
                }
                else if (dataType == deep_server::dataType::PNG) {
                    if (!cvprocess::readImage(idata, ptf_data->cv_image)) {
                        fault("Fail to read image£¡");
                        return;
                    }
                    ptf_data->input.swap(idata);
                }
                else {
                    fault("invalid data type : £¡" + dataType);
                    return;
                }

                double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                ptf_data->time_consumed.decode_time = elapsed;
                ptf_data->time_consumed.whole_time += elapsed;
                DEEP_LOG_INFO("decode image consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                if (!verifyCV_Image(ptf_data->cv_image)) {
                    fault("invalid image format £¡");
                    return;
                }

                ptf_data->t_out.info.set_width(ptf_data->cv_image.cols);
                ptf_data->t_out.info.set_height(ptf_data->cv_image.rows);
                ptf_data->t_out.info.set_channel(ptf_data->cv_image.channels());

                become(prepare_);
                send(this, prepare_atom::value, (uint64)(uint64*)ptf_data.get());
            }
            );
        }
    }
}
