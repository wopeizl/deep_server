//
// Created by Administrator on 2017/4/5.
//

#ifndef DEEP_SERVER_DEEP_SERVER_H
#define DEEP_SERVER_DEEP_SERVER_H

#define NO_STRICT 

#include "deep_config.hpp"
#include "http_parser.hpp"
#include "utils.hpp"
#include "json/json.h"
#include "cv_process.hpp"
#include "caffe_process.hpp"
#include <chrono>

typedef std::chrono::high_resolution_clock clock_;
typedef std::chrono::duration<double, std::ratio<1, 1000> > micro_second_;

//template<typename F, typename ...Args>
//static auto duration(F&& func, Args&&... args)
//{
//    auto start = std::chrono::steady_clock::now();
//    std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
//    return std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start);
//}

class httpanalyzer;

enum parser_state {
    receive_new_line,
    receive_continued_line,
    receive_second_newline_half
};

class httpanalyzer {
public:
    httpanalyzer() {
        parser_init(HTTP_BOTH);
    }

    ~httpanalyzer() {
        parser_free();
    }

    void
        parser_init(enum http_parser_type type)
    {
        num_messages = 0;

        settings.on_message_begin = message_begin_cb;
        settings.on_header_field = header_field_cb;
        settings.on_header_value = header_value_cb;
        settings.on_url = request_url_cb;
        settings.on_status = response_status_cb;
        settings.on_body = body_cb;
        settings.on_headers_complete = headers_complete_cb;
        settings.on_message_complete = message_complete_cb;
        settings.on_chunk_header = chunk_header_cb;
        settings.on_chunk_complete = chunk_complete_cb;

        parser = (http_parser*)new(http_parser);

        http_parser_init(parser, type);
    }

    void
        parser_free()
    {
        assert(parser);
        delete(parser);
        parser = NULL;
    }

    size_t parse(const char *buf, size_t len)
    {
        size_t nparsed;
        currently_parsing_eof = (len == 0);
        nparsed = http_parser_execute(parser, &settings, buf, len);
        return nparsed;
    }

    bool isDone() {
        return parser->m.message_complete_cb_called;
    }

    http_parser *parser;
private:

    int currently_parsing_eof;
    int num_messages;
    http_parser_settings settings;
};

class information {
public:
    information() :requests(0), pending_requests(0) {}

    int requests;
    int pending_requests;
};

struct http_state {
    http_state(abstract_broker* self) : self_(self) {
        // nop
    }

    ~http_state() {
        //aout(self_) << "http worker is destroyed" << endl;
    }

    abstract_broker* self_;
    httpanalyzer analyzer;
};

using input_atom = atom_constant<atom("dinput")>;
using output_atom = atom_constant<atom("doutput")>;
using framesync_atom = atom_constant<atom("dframesync")>;
using cprocess_atom = atom_constant<atom("cprocess")>;
using cpprocess_atom = atom_constant<atom("cpprocess")>;
using process_atom = atom_constant<atom("process")>;
using pprocess_atom = atom_constant<atom("pprocess")>;
using gprocess_atom = atom_constant<atom("gprocess")>;
using gpprocess_atom = atom_constant<atom("gpprocess")>;
using downloader_atom = atom_constant<atom("downloader")>;
using prepare_atom = atom_constant<atom("prepare")>;
using preprocess_atom = atom_constant<atom("preproc")>;
using fault_atom = atom_constant<atom("fault")>;

using caffe_process_atom = atom_constant<atom("cf_proc")>;
using caffe_manager_process_atom = atom_constant<atom("cfm_proc")>;


constexpr char http_valid_get[] = "GET / HTTP/1.1";
constexpr char http_valid_post[] = "POST / HTTP/1.1";

constexpr char http_get[] = "GET / HTTP/1.1\r\n"
"Host: localhost\r\n"
"Connection: close\r\n"
"Accept: text/plain\r\n"
"User-Agent: CAF/0.14\r\n"
"Accept-Language: en-US\r\n"
"\r\n";

constexpr char http_error[] = "HTTP/1.1 404 Not Found\r\n"
"Connection: close\r\n"
"\r\n";

constexpr char newline[2] = { '\r', '\n' };

constexpr const char http_ok[] = R"__(HTTP/1.1 200 OK
Content-Type: text/plain
Connection: keep-alive

d
Hi there! :)

0


)__";

//constexpr const char http_img[] = R"__(HTTP/1.1 200 OK
//Content-Type: text/html
//Connection: keep-alive
//
//
//<html>
//    <body>
//    
//<img src="data:image/png;base64,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"/> 
//</body>
//</html>
//
//
//)__";

constexpr const char http_img_header[] = R"__(HTTP/1.1 200 OK
Content-Type: json
Connection: keep-alive

)__";

constexpr const char http_img_tail[] = R"__(


)__";


template <size_t Size>
constexpr size_t cstr_size(const char(&)[Size]) {
    return Size;
}

namespace deep_server{
    using http_broker = caf::stateful_actor<http_state, broker>;
    behavior http_worker(http_broker* self, connection_handle hdl, actor cf_manager);
    behavior tcp_broker(broker* self, connection_handle hdl, actor cf_manager);

    struct time_consume{
        double whole_time;
        double jsonparse_time;
        double decode_time;
        double cvreadimage_time;
        double prepare_time;
        double preprocess_time;
        double predict_time;
        double postprocess_time;
        double writeresult_time;
    };

    struct caffe_data {
        cv::Mat cv_image;
        cv::Mat out_image;
        std::vector<boost::shared_ptr<caffe::Blob<float> >> input;
        std::vector<boost::shared_ptr<caffe::Blob<float> >> output;
        std::vector<caffe::Frcnn::BBox<float>> results;
        connection_handle handle;
        struct time_consume time_consumed;
    };

    class processor : public event_based_actor {
    public:

        ~processor() {
            DEEP_LOG_INFO("Whole time to process this image : " + boost::lexical_cast<string>(pcaffe_data->time_consumed.whole_time) + " ms!");
            DEEP_LOG_INFO("Time details  : " 
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.jsonparse_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.decode_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.cvreadimage_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.prepare_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.preprocess_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.predict_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.postprocess_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.writeresult_time) + ","
                + boost::lexical_cast<string>(pcaffe_data->time_consumed.whole_time) + ","
                + " ms!");

            DEEP_LOG_INFO("Finish this round processor.");
        }

        processor(actor_config& cfg,
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
        std::unique_ptr<caffe_data> pcaffe_data;
    };

    caf::logger& glog(actor* self);
    caf::logger& glog(broker* self);

}

#endif //DEEP_SERVER_DEEP_SERVER_H
