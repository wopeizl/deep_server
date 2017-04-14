
#include "actorlogger.hpp"
#include <boost/filesystem.hpp> 

namespace deep_server {

    namespace {

        std::string timestamp_to_d_string() {
            static char mbstr[256];
            std::time_t t = std::time(NULL);
            if (std::strftime(mbstr, sizeof(mbstr), "[%Y-%m-%d %M:%H:%S CST]", std::localtime(&t))) {
                return std::string(mbstr);
            }
            else {
                return std::to_string(make_timestamp().time_since_epoch().count());
            }
        }

        constexpr const char* log_level_name[] = {
            "ERROR",
            "WARN",
            "INFO",
            "DEBUG",
            "TRACE"
        };

        static_assert(DEEP_LOG_LEVEL >= 0 && DEEP_LOG_LEVEL <= 4,
            "assertion: 0 <= DEEP_LOG_LEVEL <= 4");

        intrusive_ptr<actorlogger> current_deep_logger;

        inline void set_current_logger(actorlogger* x) {
            current_deep_logger.reset(x);
        }

        inline actorlogger* get_current_logger() {
            return current_deep_logger.get();
        }

        void prettify_type_name(std::string& class_name) {
            //replace_all(class_name, " ", "");
            replace_all(class_name, "::", ".");
            replace_all(class_name, "(anonymousnamespace)", "ANON");
            replace_all(class_name, ".__1.", "."); // gets rid of weird Clang-lib names
                                                   // hide CAF magic in logs
            auto strip_magic = [&](const char* prefix_begin, const char* prefix_end) {
                auto last = class_name.end();
                auto i = std::search(class_name.begin(), last, prefix_begin, prefix_end);
                auto comma_or_angle_bracket = [](char c) { return c == ',' || c == '>'; };
                auto e = std::find_if(i, last, comma_or_angle_bracket);
                if (i != e) {
                    std::string substr(i + (prefix_end - prefix_begin), e);
                    class_name.swap(substr);
                }
            };
            char prefix1[] = "caf.detail.embedded<";
            strip_magic(prefix1, prefix1 + (sizeof(prefix1) - 1));
            // finally, replace any whitespace with %20
            replace_all(class_name, " ", "%20");
        }

        void prettify_type_name(std::string& class_name, const char* c_class_name) {
# if defined(CAF_LINUX) || defined(CAF_MACOS)
            int stat = 0;
            std::unique_ptr<char, decltype(free)*> real_class_name{ nullptr, free };
            auto tmp = abi::__cxa_demangle(c_class_name, nullptr, nullptr, &stat);
            real_class_name.reset(tmp);
            class_name = stat == 0 ? real_class_name.get() : c_class_name;
# else
            class_name = c_class_name;
# endif
            prettify_type_name(class_name);
        }

    } // namespace <anonymous>

    actorlogger::event::event(int l, const char* c, std::string p, std::string m)
        : next(nullptr),
        prev(nullptr),
        level(l),
        component(c),
        prefix(std::move(p)),
        msg(std::move(m)) {
        // nop
    }

    actorlogger::line_builder::line_builder() : behind_arg_(false) {
        // nop
    }

    actorlogger::line_builder& actorlogger::line_builder::operator<<(const std::string& str) {
        if (!str_.empty())
            str_ += " ";
        str_ += str;
        behind_arg_ = false;
        return *this;
    }

    actorlogger::line_builder& actorlogger::line_builder::operator<<(const char* str) {
        if (!str_.empty())
            str_ += " ";
        str_ += str;
        behind_arg_ = false;
        return *this;
    }

    std::string actorlogger::line_builder::get() const {
        return std::move(str_);
    }

    std::string actorlogger::render_type_name(const std::type_info& ti) {
        std::string result;
        prettify_type_name(result, ti.name());
        return result;
    }

    std::string actorlogger::extract_class_name(const char* prettyfun, size_t strsize) {
        auto first = prettyfun;
        // set end to beginning of arguments
        auto last = std::find(first, prettyfun + strsize, '(');
        auto jump_to_next_whitespace = [&] {
            // leave `first` unchanged if no whitespaces is present,
            // e.g., in constructor signatures
            auto tmp = std::find(first, last, ' ');
            if (tmp != last)
                first = tmp + 1;
        };
        // skip "virtual" prefix if present
        if (strncmp(prettyfun, "virtual ", std::min<size_t>(strsize, 8)) == 0)
            jump_to_next_whitespace();
        // skip return type
        jump_to_next_whitespace();
        if (first == last)
            return "";
        const char sep[] = "::"; // separator for namespaces and classes
        auto sep_first = std::begin(sep);
        auto sep_last = sep_first + 2; // using end() includes \0
        auto colons = first;
        decltype(colons) nextcs;
        while ((nextcs = std::search(colons + 1, last, sep_first, sep_last)) != last)
            colons = nextcs;
        std::string result;
        result.assign(first, colons);
        prettify_type_name(result);
        return result;
    }

    // returns the actor ID for the current thread
    actor_id actorlogger::thread_local_aid() {
        shared_lock<detail::shared_spinlock> guard{ aids_lock_ };
        auto i = aids_.find(std::this_thread::get_id());
        if (i != aids_.end())
            return i->second;
        return 0;
    }

    actor_id actorlogger::thread_local_aid(actor_id aid) {
        auto tid = std::this_thread::get_id();
        upgrade_lock<detail::shared_spinlock> guard{ aids_lock_ };
        auto i = aids_.find(tid);
        if (i != aids_.end()) {
            // we modify it despite the shared lock because the elements themselves
            // are considered thread-local
            auto res = i->second;
            i->second = aid;
            return res;
        }
        // upgrade to unique lock and insert new element
        upgrade_to_unique_lock<detail::shared_spinlock> uguard{ guard };
        aids_.emplace(tid, aid);
        return 0; // was empty before
    }

    void actorlogger::log_prefix(std::ostream& out, int level, const char* c_full_file_name,
        int line_num, const std::thread::id& tid) {
        log_prefix(out, level, "", "", "", c_full_file_name, line_num);
    }

    void actorlogger::log_prefix(std::ostream& out, int level, const char* component,
        const std::string& class_name,
        const char* function_name, const char* c_full_file_name,
        int line_num, const std::thread::id& tid) {
        std::string file_name;
        std::string full_file_name = c_full_file_name;
        auto ri = find(full_file_name.rbegin(), full_file_name.rend(), '/');
        if (ri != full_file_name.rend()) {
            auto i = ri.base();
            if (i == full_file_name.end())
                file_name = std::move(full_file_name);
            else
                file_name = std::string(i, full_file_name.end());
        }
        else {
            file_name = std::move(full_file_name);
        }
        std::ostringstream prefix;
        out << timestamp_to_d_string() <<  " ["
            << log_level_name[level] << "] "
            << class_name << " " << function_name << " "
            << file_name << ":" << line_num;
    }

    void actorlogger::log(int level, const char* component,
        const std::string& class_name, const char* function_name,
        const char* c_full_file_name, int line_num,
        const std::string& msg) {
        CAF_ASSERT(level >= 0 && level <= 4);
        if (level > level_)
            return;
        std::ostringstream prefix;
        log_prefix(prefix, level, component, class_name, function_name,
            c_full_file_name, line_num);
        queue_.synchronized_enqueue(queue_mtx_, queue_cv_,
            new event{level, component, prefix.str(), msg});
    }

    actorlogger* actorlogger::current_logger() {
        return get_current_logger();
    }

    void actorlogger::log_static(int level, const char* component,
        const std::string& class_name,
        const char* function_name, const char* file_name,
        int line_num, const std::string& msg) {
        auto ptr = get_current_logger();
        if (ptr != nullptr)
            ptr->log(level, component, class_name, function_name, file_name, line_num,
                msg);
    }

    actorlogger::~actorlogger() {
        stop();
        // tell system our dtor is done
        //std::unique_lock<std::mutex> guard{ system_.logger_dtor_mtx_ };
        //system_.logger_dtor_done_ = true;
        //system_.logger_dtor_cv_.notify_one();
    }

    void actorlogger::set_current_actor_logger(actorlogger* x) {
        set_current_logger(x);
    }

    actorlogger::actorlogger(actor_system& sys, deepconfig& cfg) : system_(sys), config_(cfg){
        CAF_IGNORE_UNUSED(cfg);
        auto lvl_atom = cfg.sys_log_level;
        switch (static_cast<uint64_t>(lvl_atom)) {
        case error_log_lvl_atom::uint_value():
            level_ = DEEP_LOG_LEVEL_ERROR;
            break;
        case warning_log_lvl_atom::uint_value():
            level_ = DEEP_LOG_LEVEL_WARNING;
            break;
        case info_log_lvl_atom::uint_value():
            level_ = DEEP_LOG_LEVEL_INFO;
            break;
        case debug_log_lvl_atom::uint_value():
            level_ = DEEP_LOG_LEVEL_DEBUG;
            break;
        case trace_log_lvl_atom::uint_value():
            level_ = DEEP_LOG_LEVEL_TRACE;
            break;
        default: {
            level_ = DEEP_LOG_LEVEL;
        }
        }
    }

    void actorlogger::run() {
        std::fstream file;
        namespace fs = boost::filesystem;
        if (config_.sys_log_filename.empty()) {
            std::cout << "Please give the sys log file in config file, server will run without log output." << endl;
                return;
        }
        else {
            auto f = config_.sys_log_filename;
            // Replace placeholders.
            //const char pid[] = "[PID]";
            //auto i = std::search(f.begin(), f.end(), std::begin(pid),
            //    std::end(pid) - 1);
            //if (i != f.end()) {
            //    auto id = std::to_string(detail::get_process_id());
            //    f.replace(i, i + sizeof(pid) - 1, id);
            //}
            const char ts[] = "[TIMESTAMP]";
            auto i = std::search(f.begin(), f.end(), std::begin(ts), std::end(ts) - 1);
            if (i != f.end()) {
                auto now = timestamp_to_string(make_timestamp());
                f.replace(i, i + sizeof(ts) - 1, now);
            }
            //const char node[] = "[NODE]";
            //i = std::search(f.begin(), f.end(), std::begin(node), std::end(node) - 1);
            //if (i != f.end()) {
            //    auto nid = to_string(system_.node());
            //    f.replace(i, i + sizeof(node) - 1, nid);
            //}
            file.open(f, std::ios::out | std::ios::app);
            if (!file) {
                std::cerr << "unable to open log file " << f << std::endl;
                return;
            }
        }
        if (file) {
            // log first entry
            constexpr atom_value level_names[] = {
                error_log_lvl_atom::value, warning_log_lvl_atom::value,
                info_log_lvl_atom::value, debug_log_lvl_atom::value,
                trace_log_lvl_atom::value };
            auto lvl_atom = level_names[DEEP_LOG_LEVEL];
            //log_prefix(file, DEEP_LOG_LEVEL_INFO, "caf", "caf.logger", "start",
            //    __FILE__, __LINE__, parent_thread_);
            file << " level = " << to_string(lvl_atom) << ", node = "
                << to_string(system_.node()) << std::endl;
        }
        // receive log entries from other threads and actors
        std::unique_ptr<event> ptr;
        for (;;) {
            // make sure we have data to read
            queue_.synchronized_await(queue_mtx_, queue_cv_);
            // read & process event
            ptr.reset(queue_.try_pop());
            CAF_ASSERT(ptr != nullptr);
            // empty message means: shut down
            //if (ptr->msg.empty())
            //    break;

            if (file && !ptr->msg.empty())
                file << ptr->prefix << ' ' << ptr->msg << std::endl;
        }
        if (file) {
            log_prefix(file, DEEP_LOG_LEVEL_INFO, 
                __FILE__, __LINE__, parent_thread_);
            file << " EOF" << std::endl;
        }
    }

    void actorlogger::start() {
        parent_thread_ = std::this_thread::get_id();
        if (config_.sys_log_level == quiet_log_lvl_atom::uint_value())
            return;
        thread_ = std::thread{ [this] { this->run(); } };
    }

    void actorlogger::stop() {
        if (!thread_.joinable())
            return;
        // an empty string means: shut down
        queue_.synchronized_enqueue(queue_mtx_, queue_cv_, new event);
        thread_.join();
    }

} // namespace caf
