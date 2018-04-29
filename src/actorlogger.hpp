#pragma once

#include "deep_config.hpp"

namespace deep_server{

    enum backend_mode_t {
        caffe = 0,
        yolo,
        tensorflow
    };

    struct lib_mode_t {
        bool caffe;
        bool yolo;
        bool tensorflow;
        lib_mode_t():caffe(false), yolo(false), tensorflow(false){}
    };

    void setlibmode(string mode);
    lib_mode_t getlibmode();

    class deepconfig : public actor_system_config {
    public:
        uint16_t port = 12345;
        std::string host = "0.0.0.0";
        bool server_mode = true;
        bool http_mode = true;
        //using sys_log_level = trace_log_lvl_atom;
        string sys_log_filename = "";
        uint64_t sys_log_level = trace_log_lvl_atom::uint_value();
        bool pb_log_level = true;
        string caffe_model = "";
        string caffe_weights = "";
        string caffe_default_c = "";
        string gpu = "";
        string caffe_gpu = "";
        string yolo_gpu = "";
        string tf_gpu = "";
        string cpu = "";
        int batch_size = 1;
        string lib_mode = "caffe";
        string yolo_cfgfile;
        string yolo_weightfile;
        string yolo_namelist;
        string yolo_labeldir;
        float yolo_thresh;
        float yolo_hier_thresh;
        string tf_model_name = "";
        string tf_model_base_path = "";
        float tf_thres = 0.5;
        int tf_input_height = 384;
        int tf_input_width  = 1248;
        bool tf_detect = true;

        deepconfig() {
            opt_group{ custom_options_, "global" }
                .add(port, "port,p", "set port")
                .add(host, "host,H", "set host (ignored in server mode)")
                .add(server_mode, "server-mode,s", "enable server mode")
                .add(sys_log_filename, "sys-log,l", "system log file")
                .add(caffe_model, "caffe_model", "caffe model file")
                .add(caffe_weights, "caffe_weights", "caffe weights file")
                .add(caffe_default_c, "caffe_default_c", "caffe default_c file")
                .add(gpu, "gpu,g", "gpu numbers, notes that this option will disable the cpu option")
                .add(caffe_gpu, "caffe_gpu", "caffe_gpu numbers, notes that this option will disable the cpu option")
                .add(yolo_gpu, "yolo_gpu", "yolo_gpu numbers, notes that this option will disable the cpu option")
                .add(tf_gpu, "tf_gpu", "tf_gpu numbers, notes that this option will disable the cpu option")
                .add(cpu, "cpu,c", "cpu numbers")
                .add(batch_size, "batch_size,b", "lib process pics batch size numbers")
                .add(lib_mode, "lib_mode", "supported libraries, for now it is caffe and yolo")
                .add(yolo_cfgfile, "yolo_cfgfile", "yolo cfgfile")
                .add(yolo_weightfile, "yolo_weightfile", "yolo weightfile")
                .add(yolo_namelist, "yolo_namelist", "yolo namelist")
                .add(yolo_labeldir, "yolo_labeldir", "yolo labeldir")
                .add(yolo_thresh, "yolo_thresh", "yolo thresh")
                .add(yolo_hier_thresh, "yolo_hier_thresh", "yolo_hier_thresh")
                .add(tf_model_base_path, "tf_model_base_path", "tensorflow model_base_path")
                .add(tf_model_name, "tf_model_name", "name of model of tensorflow")
                .add(tf_thres, "tf_thres", "tensorflow threshold")
                .add(tf_input_height, "tf_input_height", "tensorflow image height")
                .add(tf_input_width, "tf_input_width", "tensorflow image width")
                .add(tf_detect, "tf_detect", "tensorflow detect model")
                .add(http_mode, "http-mode,hm", "enable http mode");
        }

        deepconfig& operator=(const deepconfig&) = default;
    };

/// Centrally logs events from all actors in an actor system. To enable
/// logging in your application, you need to define `DEEP_LOG_LEVEL`. Per
/// default, the logger generates log4j compatible output.
class actorlogger : public ref_counted {
public:
    friend class actor_system;

    struct event {
        event* next;
        event* prev;
        int level;
        const char* component;
        std::string prefix;
        std::string msg;
        explicit event(int l = 0, char const* c = "", std::string p = "",
            std::string m = "");
    };

    template <class T>
    struct arg_wrapper {
        const char* name;
        const T& value;
        arg_wrapper(const char* x, const T& y) : name(x), value(y) {
            // nop
        }
    };

    template <class T>
    static arg_wrapper<T> make_arg_wrapper(const char* name, const T& value) {
        return{ name, value };
    }

    static std::string render_type_name(const std::type_info& ti);

    class line_builder {
    public:
        line_builder();

        template <class T>
        line_builder& operator<<(const T& x) {
            if (!str_.empty())
                str_ += " ";
            str_ += deep_to_string(x);
            behind_arg_ = false;
            return *this;
        }

        template <class T>
        line_builder& operator<<(const arg_wrapper<T>& x) {
            if (behind_arg_)
                str_ += ", ";
            else if (!str_.empty())
                str_ += " ";
            str_ += x.name;
            str_ += " = ";
            str_ += deep_to_string(x.value);
            behind_arg_ = true;
            return *this;
        }

        line_builder& operator<<(const std::string& str);

        line_builder& operator<<(const char* str);

        std::string get() const;

    private:
        std::string str_;
        bool behind_arg_;
    };

    static std::string extract_class_name(const char* prettyfun, size_t strsize);

    template <size_t N>
    static std::string extract_class_name(const char(&prettyfun)[N]) {
        return extract_class_name(prettyfun, N);
    }

    /// Returns the ID of the actor currently associated to the calling thread.
    actor_id thread_local_aid();

    /// Associates an actor ID to the calling thread and returns the last value.
    actor_id thread_local_aid(actor_id aid);

    /// Writes an entry to the log file.
    void log(int level, const char* component, const std::string& class_name,
        const char* function_name, const char* c_full_file_name,
        int line_num, const std::string& msg);

    ~actorlogger() override;

    /** @cond PRIVATE */

    static void set_current_actor_logger(actorlogger*);

    static actorlogger* current_logger();

    static void log_static(int level, const char* component,
        const std::string& class_name,
        const char* function_name,
        const char* file_name, int line_num,
        const std::string& msg);

    /** @endcond */
    actorlogger(actor_system& sys, deepconfig& cfg);

    void start();

    void stop();

private:

    void run();

    void log_prefix(std::ostream& out, int level, const char* c_full_file_name,
        int line_num, const std::thread::id& tid = std::this_thread::get_id());

    void log_prefix(std::ostream& out, int level, const char* component,
        const std::string& class_name, const char* function_name,
        const char* c_full_file_name, int line_num,
        const std::thread::id& tid = std::this_thread::get_id());

    actor_system& system_;
    int level_;
    caf::detail::shared_spinlock aids_lock_;
    std::unordered_map<std::thread::id, actor_id> aids_;
    std::thread thread_;
    std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    caf::detail::single_reader_queue<event> queue_;
    std::thread::id parent_thread_;
    deepconfig& config_;
};

#define DEEP_VOID_STMT static_cast<void>(0)

#define CAF_CAT(a, b) a##b

#define DEEP_LOG_LEVEL_ERROR 0
#define DEEP_LOG_LEVEL_WARNING 1
#define DEEP_LOG_LEVEL_INFO 2
#define DEEP_LOG_LEVEL_DEBUG 3
#define DEEP_LOG_LEVEL_TRACE 4

#define DEEP_LOG_LEVEL DEEP_LOG_LEVEL_TRACE

#define DEEP_ARG(argument) deep_server::actorlogger::make_arg_wrapper(#argument, argument)

#define DEEP_GET_CLASS_NAME  ""
#ifdef CAF_MSVC
//#define DEEP_GET_CLASS_NAME deep_server::actorlogger::extract_class_name(__FUNCSIG__)
#else // CAF_MSVC
//#define DEEP_GET_CLASS_NAME deep_server::actorlogger::extract_class_name(__PRETTY_FUNCTION__)
#endif // CAF_MSVC

#define DEEP_SET_LOGGER_SYS deep_server::actorlogger::set_current_actor_logger(ptr)

#ifndef DEEP_LOG_LEVEL

#define DEEP_LOG_IMPL(unused1, unused2, unused3)

// placeholder macros when compiling without logging
inline caf::actor_id DEEP_SET_AID_dummy() { return 0; }

#define DEEP_PUSH_AID(unused) DEEP_VOID_STMT

#define DEEP_PUSH_AID_FROM_PTR(unused) DEEP_VOID_STMT

#define DEEP_SET_AID(unused) DEEP_SET_AID_dummy()

#define DEEP_LOG_TRACE(unused)

#else // DEEP_LOG_LEVEL

#ifndef DEEP_LOG_COMPONENT
#define DEEP_LOG_COMPONENT "deep_server"
#endif

#define DEEP_LOG_IMPL(component, loglvl, message)                               \
  deep_server::actorlogger::log_static(loglvl, component, DEEP_GET_CLASS_NAME,               \
                          "", __FILE__, __LINE__,                        \
                          (deep_server::actorlogger::line_builder{} << message).get())

#define DEEP_PUSH_AID(aarg)                                                     \
  auto DEEP_UNIFYN(caf_tmp_ptr) = deep_server::actorlogger::current_logger();                \
  caf::actor_id DEEP_UNIFYN(caf_aid_tmp) = 0;                                   \
  if (DEEP_UNIFYN(caf_tmp_ptr))                                                 \
    DEEP_UNIFYN(caf_aid_tmp) = DEEP_UNIFYN(caf_tmp_ptr)->thread_local_aid(aarg); \
  auto DEEP_UNIFYN(aid_aid_tmp_guard) = caf::detail::make_scope_guard([=] {     \
    auto DEEP_UNIFYN(caf_tmp2_ptr) = deep_server::actorlogger::current_logger();             \
    if (DEEP_UNIFYN(caf_tmp2_ptr))                                              \
      DEEP_UNIFYN(caf_tmp2_ptr)->thread_local_aid(DEEP_UNIFYN(caf_aid_tmp));     \
  })

#define DEEP_PUSH_AID_FROM_PTR(some_ptr)                                        \
  auto DEEP_UNIFYN(caf_aid_ptr) = some_ptr;                                     \
  DEEP_PUSH_AID(DEEP_UNIFYN(caf_aid_ptr) ? DEEP_UNIFYN(caf_aid_ptr)->id() : 0)

#define DEEP_SET_AID(aid_arg)                                                   \
  (deep_server::actorlogger::current_logger()                                               \
     ? deep_server::actorlogger::current_logger()->thread_local_aid(aid_arg)                \
     : 0)


#if DEEP_LOG_LEVEL < DEEP_LOG_LEVEL_TRACE

#define DEEP_LOG_TRACE(unused) DEEP_VOID_STMT

#else // DEEP_LOG_LEVEL < DEEP_LOG_LEVEL_TRACE

#define DEEP_LOG_TRACE(entry_message)                                           \
  const char* DEEP_UNIFYN(func_name_) = __func__;                               \
  DEEP_LOG_IMPL(DEEP_LOG_COMPONENT, DEEP_LOG_LEVEL_TRACE,                         \
               "ENTRY" << entry_message);                                      \
  auto DEEP_UNIFYN(DEEP_LOG_trace_guard_) = ::caf::detail::make_scope_guard([=] {\
    deep_server::actorlogger::log_static(DEEP_LOG_LEVEL_TRACE, DEEP_LOG_COMPONENT,            \
                            DEEP_GET_CLASS_NAME, DEEP_UNIFYN(func_name_),        \
                            __FILE__, __LINE__, "EXIT");                       \
  })

#endif // DEEP_LOG_LEVEL < DEEP_LOG_LEVEL_TRACE

#if DEEP_LOG_LEVEL >= DEEP_LOG_LEVEL_DEBUG
#  define DEEP_LOG_DEBUG(output)                                                \
     DEEP_LOG_IMPL(DEEP_LOG_COMPONENT, DEEP_LOG_LEVEL_DEBUG, output)
#endif

#if DEEP_LOG_LEVEL >= DEEP_LOG_LEVEL_INFO
#  define DEEP_LOG_INFO(output)                                                 \
     DEEP_LOG_IMPL(DEEP_LOG_COMPONENT, DEEP_LOG_LEVEL_INFO, output)
#endif

#if DEEP_LOG_LEVEL >= DEEP_LOG_LEVEL_WARNING
#  define DEEP_LOG_WARNING(output)                                              \
     DEEP_LOG_IMPL(DEEP_LOG_COMPONENT, DEEP_LOG_LEVEL_WARNING, output)
#endif

#define DEEP_LOG_ERROR(output)                                                  \
  DEEP_LOG_IMPL(DEEP_LOG_COMPONENT, DEEP_LOG_LEVEL_ERROR, output)

#endif // DEEP_LOG_LEVEL

#ifndef DEEP_LOG_INFO
#  define DEEP_LOG_INFO(output) DEEP_VOID_STMT
#endif

#ifndef DEEP_LOG_DEBUG
#  define DEEP_LOG_DEBUG(output) DEEP_VOID_STMT
#endif

#ifndef DEEP_LOG_WARNING
#  define DEEP_LOG_WARNING(output) DEEP_VOID_STMT
#endif

#ifndef DEEP_LOG_ERROR
#  define DEEP_LOG_ERROR(output) DEEP_VOID_STMT
#endif

#ifdef DEEP_LOG_LEVEL

#define DEEP_LOG_DEBUG_IF(cond, output)                                         \
  if (cond) { DEEP_LOG_IMPL(DEEP_LOG_COMPONENT, DEEP_LOG_LEVEL_DEBUG, output); }  \
    DEEP_VOID_STMT

#define DEEP_LOG_INFO_IF(cond, output)                                          \
  if (cond) { DEEP_LOG_IMPL(DEEP_LOG_COMPONENT, DEEP_LOG_LEVEL_INFO, output); }   \
    DEEP_VOID_STMT

#define DEEP_LOG_WARNING_IF(cond, output)                                       \
  if (cond) { DEEP_LOG_IMPL(DEEP_LOG_COMPONENT, DEEP_LOG_LEVEL_WARNING, output); }\
    DEEP_VOID_STMT

#define DEEP_LOG_ERROR_IF(cond, output)                                         \
  if (cond) { DEEP_LOG_IMPL(DEEP_LOG_COMPONENT, DEEP_LOG_LEVEL_ERROR, output); }  \
    DEEP_VOID_STMT

#else // DEEP_LOG_LEVEL

#define DEEP_LOG_DEBUG_IF(unused1, unused2)
#define DEEP_LOG_INFO_IF(unused1, unused2)
#define DEEP_LOG_WARNING_IF(unused1, unused2)
#define DEEP_LOG_ERROR_IF(unused1, unused2)

#endif // DEEP_LOG_LEVEL

// -- Standardized CAF events according to SE-0001.

#define DEEP_LOG_EVENT_COMPONENT "se-0001"

#define DEEP_LOG_SPAWN_EVENT(aid, aargs)                                        \
  DEEP_LOG_IMPL(DEEP_LOG_EVENT_COMPONENT, DEEP_LOG_LEVEL_DEBUG,                   \
               "SPAWN ; ID =" << aid                                           \
               << "; ARGS =" << deep_to_string(aargs).c_str())

#define DEEP_LOG_INIT_EVENT(aName, aHide)                                       \
  DEEP_LOG_IMPL(DEEP_LOG_EVENT_COMPONENT, DEEP_LOG_LEVEL_DEBUG,                   \
               "INIT ; NAME =" << aName << "; HIDDEN =" << aHide)

/// Logs
#define DEEP_LOG_SEND_EVENT(ptr)                                                \
  DEEP_LOG_IMPL(DEEP_LOG_EVENT_COMPONENT, DEEP_LOG_LEVEL_DEBUG,                   \
               "SEND ; TO ="                                                   \
               << deep_to_string(strong_actor_ptr{this->ctrl()}).c_str()       \
               << "; FROM =" << deep_to_string(ptr->sender).c_str()            \
               << "; STAGES =" << deep_to_string(ptr->stages).c_str()          \
               << "; CONTENT =" << deep_to_string(ptr->content()).c_str())

#define DEEP_LOG_RECEIVE_EVENT(ptr)                                             \
  DEEP_LOG_IMPL(DEEP_LOG_EVENT_COMPONENT, DEEP_LOG_LEVEL_DEBUG,                   \
               "RECEIVE ; FROM ="                                              \
               << deep_to_string(ptr->sender).c_str()                          \
               << "; STAGES =" << deep_to_string(ptr->stages).c_str()          \
               << "; CONTENT =" << deep_to_string(ptr->content()).c_str())

#define DEEP_LOG_REJECT_EVENT()                                                 \
  DEEP_LOG_IMPL(DEEP_LOG_EVENT_COMPONENT, DEEP_LOG_LEVEL_DEBUG, "REJECT")

#define DEEP_LOG_ACCEPT_EVENT()                                                 \
  DEEP_LOG_IMPL(DEEP_LOG_EVENT_COMPONENT, DEEP_LOG_LEVEL_DEBUG, "ACCEPT")

#define DEEP_LOG_DROP_EVENT()                                                   \
  DEEP_LOG_IMPL(DEEP_LOG_EVENT_COMPONENT, DEEP_LOG_LEVEL_DEBUG, "DROP")

#define DEEP_LOG_SKIP_EVENT()                                                   \
  DEEP_LOG_IMPL(DEEP_LOG_EVENT_COMPONENT, DEEP_LOG_LEVEL_DEBUG, "SKIP")

#define DEEP_LOG_FINALIZE_EVENT()                                               \
  DEEP_LOG_IMPL(DEEP_LOG_EVENT_COMPONENT, DEEP_LOG_LEVEL_DEBUG, "FINALIZE")

#define DEEP_LOG_TERMINATE_EVENT(rsn)                                           \
  DEEP_LOG_IMPL(DEEP_LOG_EVENT_COMPONENT, DEEP_LOG_LEVEL_DEBUG,                   \
               "TERMINATE ; REASON =" << deep_to_string(rsn).c_str())

}