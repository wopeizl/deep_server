
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

#if defined WIN32 || defined WINCE
#else

    yolo_init_net(gpu_id, cfgfile, weightfile, namelist, labeldir, thresh, hier_thresh);

#endif

    return true;
}

#if defined WIN32 || defined WINCE
#else

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
    yolo_predict(input, &out);

    return true;
}
#endif

namespace deep_server {

    yolo_processor::yolo_processor(actor_config& cfg,
        std::string  n, actor s, connection_handle handle, actor cm)
        : event_based_actor(cfg),
        name_(std::move(n)), bk(s), handle_(handle), cf_manager(cm) {

        pyolo_data.reset(new yolo_data());
        pyolo_data->time_consumed.whole_time = 0.f;
        pyolo_data->self = this;
        //set_default_handler(skip);

        prepare_.assign(
            [=](prepare_atom, uint64 datapointer) {

            std::chrono::time_point<clock_> beg_ = clock_::now();

#if defined WIN32 || defined WINCE
#else
            if (!yolo_process::prepare(pyolo_data->cv_image, &pyolo_data->input)) {
                fault("Fail to prepare image£¡");
                return;
            }
#endif

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
            send(this, downloader_atom::value, handle, odata);
        }
        );
        downloader_.assign(
            [=](downloader_atom, connection_handle handle, vector<unsigned char>& odata) {

            DEEP_LOG_INFO(name_ + " starts to downloader_atom.");

            if (!cvprocess::writeImage(pyolo_data->output, odata)) {
                fault("fail to cvprocess::process_caffe_result ! ");
                return;
            }

            send(bk, handle, output_atom::value, base64_encode(odata.data(), odata.size()));
        }
        );
    }

    behavior yolo_processor::make_behavior() {
        //send(this, input_atom::value);
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

            std::string method = value.get("method", "").asString();
            if (method == "flip") {
                std::vector<unsigned char> odata;
                if (!cvprocess::flip(idata, odata)) {
                    fault("Fail to process pic by " + method + ", please check£¡");
                    return;
                }

                become(downloader_);
                send(this, downloader_atom::value, handle_, odata);
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

                become(prepare_);
                send(this, prepare_atom::value, (uint64)(uint64*)pyolo_data.get());
            }
            else {
                fault("invalid method : £¡" + method);
                return;
            }
        }
        );
    }
}