//
// Created by Administrator on 2017/4/5.
//

#ifndef DEEP_SERVER_DEEP_SERVER_H
#define DEEP_SERVER_DEEP_SERVER_H

#include "caf/config.hpp"

#ifdef WIN32
# define _WIN32_WINNT 0x0600
# include <Winsock2.h>
#else
# include <arpa/inet.h> // htonl
#endif

#include <vector>
#include <string>
#include <limits>
#include <memory>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <csignal>
#include <ctime>

#include "caf/all.hpp"
#include "caf/io/all.hpp"

using std::cout;
using std::cerr;
using std::endl;

using namespace caf;
using namespace caf::io;
using namespace std;

#ifndef UNUSED
#define UNUSED(expr) do { (void)(expr); } while (0)
#endif
#ifdef __clang__
# pragma clang diagnostic ignored "-Wshorten-64-to-32"
# pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
# pragma clang diagnostic ignored "-Wunused-const-variable"
# pragma clang diagnostic ignored "-Wunused-parameter"

#endif // __clang__


enum parser_state {
    receive_new_line,
    receive_continued_line,
    receive_second_newline_half
};

struct http_state {
    http_state(abstract_broker* self) : self_(self) {
        // nop
    }

    ~http_state() {
        aout(self_) << "http worker is destroyed";
    }

    std::vector<std::string> lines;
    parser_state ps = receive_new_line;
    abstract_broker* self_;
};

namespace deep_server{
    using http_broker = caf::stateful_actor<http_state, broker>;
    behavior http_worker(http_broker* self, connection_handle hdl);
}

#endif //DEEP_SERVER_DEEP_SERVER_H
