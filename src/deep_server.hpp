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
#include <chrono>
#include "deep.pb.h"

typedef std::chrono::high_resolution_clock clock_;
typedef std::chrono::duration<double, std::ratio<1, 1000> > mill_second_;

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

using caffe_process_atom = atom_constant<atom("caffe_proc")>;
using yolo_process_atom = atom_constant<atom("yolo_proc")>;
using lib_preprocess_atom = atom_constant<atom("cf_preproc")>;
using deep_manager_process_atom = atom_constant<atom("cf_manager")>;

constexpr const char http_img_header[] = R"__(HTTP/1.1 200 OK\r\n)__";
constexpr const char http_img_tail[] = R"__(\r\n)__";

template <size_t Size>
constexpr size_t cstr_size(const char(&)[Size]) {
    return Size;
}

namespace deep_server{
    using http_broker = caf::stateful_actor<http_state, broker>;
    behavior http_worker(http_broker* self, connection_handle hdl, actor cf_manager);
    void tcp_broker(broker* self, connection_handle hdl, actor cf_manager);

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

    caf::logger& glog(actor* self);
    caf::logger& glog(broker* self);

}

#endif //DEEP_SERVER_DEEP_SERVER_H
