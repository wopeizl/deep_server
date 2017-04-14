
#include "deep_server.hpp"
#include "nlogger.hpp"
#include "utils.hpp"

static constexpr const char base64_tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"abcdefghijklmnopqrstuvwxyz"
"0123456789+/";

std::string encode_base64(const string& str) {
    std::string result;
    // consumes three characters from input
    auto consume = [&](const char* i) {
        int buf[]{
            (i[0] & 0xfc) >> 2,
            ((i[0] & 0x03) << 4) + ((i[1] & 0xf0) >> 4),
            ((i[1] & 0x0f) << 2) + ((i[2] & 0xc0) >> 6),
            i[2] & 0x3f
        };
        for (auto x : buf)
            result += base64_tbl[x];
    };
    // iterate string in chunks of three characters
    auto i = str.begin();
    for (; std::distance(i, str.end()) >= 3; i += 3)
        consume(&(*i));
    if (i != str.end()) {
        // "fill" string with 0s
        char cbuf[] = { 0, 0, 0 };
        std::copy(i, str.end(), cbuf);
        consume(cbuf);
        // override filled characters (garbage) with '='
        for (auto j = result.end() - (3 - (str.size() % 3)); j != result.end(); ++j)
            *j = '=';
    }
    return result;
}

namespace deep_server {

    static caf::logger* _logger = nullptr;
    atomic_flag flag = ATOMIC_FLAG_INIT;

    caf::logger& _glog(actor_system& system) {
        if (_logger == nullptr) {
            while (flag.test_and_set());
            if (_logger == nullptr) {
                _logger = &system.logger();
            }
            flag.clear();
        }
        return *_logger;
    }
    caf::logger& glog(actor* self) {
        if (_logger == nullptr) {
            return _glog(self->home_system());
        }
        return *_logger;
    }
    caf::logger& glog(broker* self) {
        if (_logger == nullptr) {
            return _glog(self->home_system());
        }
        return *_logger;
    }

    const char* fill(size_t line) {
        if (line < 10)
            return "    ";
        if (line < 100)
            return "   ";
        if (line < 1000)
            return "  ";
        return " ";
    }

    void remove_trailing_spaces(std::string& x) {
        x.erase(std::find_if_not(x.rbegin(), x.rend(), ::isspace).base(), x.end());
    }


    nlogger::stream::stream(nlogger& parent, nlogger::level lvl)
        : parent_(parent),
        lvl_(lvl) {
        // nop
    }

    bool nlogger::init(int lvl_cons, int lvl_file, const std::string& logfile) {
        instance().level_console_ = static_cast<level>(lvl_cons);
        instance().level_file_ = static_cast<level>(lvl_file);
        if (!logfile.empty()) {
            instance().file_.open(logfile, std::ofstream::out | std::ofstream::app);
            return !!instance().file_;
        }
        return true;
    }

    nlogger& nlogger::instance() {
        static nlogger l;
        return l;
    }

    nlogger::stream nlogger::error() {
        return stream{ *this, level::error };
    }

    nlogger::stream nlogger::info() {
        return stream{ *this, level::info };
    }

    nlogger::stream nlogger::verbose() {
        return stream{ *this, level::verbose };
    }

    nlogger::stream nlogger::massive() {
        return stream{ *this, level::massive };
    }

    void nlogger::disable_colors() {
        // Disable colors by piping all writes through dummy_.
        // Since dummy_ is not a TTY, colors are turned off implicitly.
        dummy_.copyfmt(std::cerr);
        dummy_.clear(std::cerr.rdstate());
        dummy_.basic_ios<char>::rdbuf(std::cerr.rdbuf());
        console_ = &dummy_;
    }

    nlogger::nlogger()
        : level_console_(level::error),
        level_file_(level::error),
        console_(&std::cerr) {
        // nop
    }

}