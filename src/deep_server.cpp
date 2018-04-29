
#include "color.h"
#include "caffe_process.hpp"
#include "yolo_process.hpp"
#include "tf_process.hpp"

namespace deep_server{

    static lib_mode_t g_lib_mode;

    void setlibmode(string mode) {
        vector<string> modes;
        boost::split(modes, mode, boost::is_any_of(","));

        for (auto iter = modes.begin(); iter != modes.end(); ++iter) {
            if (*iter == "caffe")
                g_lib_mode.caffe = true;
            else if (*iter == "yolo")
                g_lib_mode.yolo = true;
            else if (*iter == "tensorflow")
                g_lib_mode.tensorflow = true;
        }
    }

    lib_mode_t getlibmode() {
        return g_lib_mode;
    }

    bool verifyCV_Image(const cv::Mat& mat, int maxh, int maxw, int maxc, int minh, int minw, int minc) {
        if (!mat.empty()) {
            if (mat.data != nullptr) {
                if (mat.rows >= minh && mat.rows <= maxh
                    &&mat.cols >= minw && mat.cols <= maxw
                    &&mat.channels() >= minc && mat.channels() <= maxc
                    ) {
                    return true;
                }
            }
        }

        return false;
    }

    // utility function to print an exit message with custom name
    void print_on_exit(const actor& hdl, const std::string& name) {
        hdl->attach_functor([=](const caf::error& reason) {
            DEEP_LOG_INFO(name + " exited: " + to_string(reason));
        });
    }

    struct base_package{};
    struct caffe_batch_package : base_package {
        vector<caffe_data*> sources;
        std::vector<cv::Mat > imgs;
        std::vector<boost::shared_ptr<caffe::Blob<float> >> input;
        std::vector<boost::shared_ptr<caffe::Blob<float> >> output;
    };

    struct yolo_batch_package : base_package {
        vector<yolo_data*> sources;
        image* input;
        cv::Mat output;

        yolo_batch_package():input(nullptr){}
        ~yolo_batch_package() {
            if (input) {
            //    delete input;
            }
        }
    };

    struct tf_batch_package : base_package {
        vector<tf_data*> sources;
        std::vector<cv::Mat > imgs;
        std::vector<vector<unsigned char> > idatas;
        std::vector<c_tf_detect_result> detect_output;
        std::vector<c_tf_classify_result> classify_output;
    };

    class tf_proc : public blocking_actor {
    public:
        tf_proc(actor_config& cfg, int count, int i, int bz) : blocking_actor(cfg), xcfg(cfg), solver_count(count), index(i), batch_size(bz) {
        }

        void act() override {
            vector<string> flags;
            deepconfig& dcfg = (deepconfig&)xcfg.host->system().config();
            flags.push_back("");
            flags.push_back("--graph=" + dcfg.tf_model_base_path);
            flags.push_back("--dev_list=" + dcfg.tf_gpu);
            flags.push_back("--thres_hold=" + boost::lexical_cast<string>(dcfg.tf_thres));
            flags.push_back("--input_width=" + boost::lexical_cast<string>(dcfg.tf_input_width));
            flags.push_back("--input_height=" + boost::lexical_cast<string>(dcfg.tf_input_height));
            bool is_detect = dcfg.tf_detect;
            //if (index >= 0) {
            //    flags.push_back("--gpu");
            //    flags.push_back(boost::lexical_cast<string, int>(index));
            //}
            //else {
            //    flags.push_back("--cpu");
            //    flags.push_back("-1");
            //}

            std::chrono::time_point<clock_> beg_ = clock_::now();
            if (!tfp.init(flags, index)) {
                    DEEP_LOG_ERROR("failed to start tensorflow !!!!!!!!!!!!!!!!!!!!!!! ");
            }
            double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
            DEEP_LOG_INFO("tensorflow initialization consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

            bool running = true;
            this->receive_while(running) (
                [=](const tf_process_atom, uint64 datapointer) {
                DEEP_LOG_INFO("new request to tensorflow proc " + boost::lexical_cast<string>(index) + " to process data");
                tf_batch_package *data = (tf_batch_package*)datapointer;
                std::chrono::time_point<clock_> beg_ = clock_::now();
                if (is_detect) {
                    tfp.detect_predict(data->imgs[0], data->detect_output);
                }
                else {
                    tfp.classify_predict(data->imgs[0], data->classify_output);
                }
                double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                DEEP_LOG_INFO("tensorflow predict " + boost::lexical_cast<string>(data->sources.size()) + " pics one time consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");
                for (int i = 0; i < data->sources.size(); ++i) {
                    //actor back = actor_cast<actor>(data->sources[i]->self);
                    actor& back = data->sources[i]->self;
                    data->sources[i]->time_consumed.predict_time = elapsed;
                    data->sources[i]->time_consumed.whole_time += elapsed;

                    if (is_detect) {
                        tfp.postprocess(data->idatas, i, data->detect_output, data->sources[i]->detect_results);
                    }
                    else {
                        tfp.postprocess(data->idatas, i, data->classify_output, data->sources[i]->classify_results);
                    }
                    this->send(back, process_atom::value, (uint64)0);
                }
                delete data;
            },
                [=](tick_atom, size_t interval) {
                delayed_send(this, std::chrono::milliseconds{ interval },
                    tick_atom::value, interval);
            },
                [=](vector <std::pair<uint64, uint64> > a) {
            },
                [&](exit_msg& em) {
                if (em.reason) {
                    this->fail_state(std::move(em.reason));
                    running = false;
                }
            }
            );
        }

    private:
        tf_process tfp;
        actor_config xcfg;
        int solver_count;
        int batch_size;
        int index;
        vector <std::pair<uint64, uint64> > batch_tasks;
    };

    class caffe_proc : public blocking_actor{
    public:
        caffe_proc(actor_config& cfg, int count, int i, int bz) : blocking_actor(cfg), xcfg(cfg), solver_count(count), index(i), batch_size(bz) {
        }

        void act() override {
            vector<string> flags;
            deepconfig& dcfg = (deepconfig&)xcfg.host->system().config();
            flags.push_back("");
            flags.push_back("--model");
            flags.push_back(dcfg.caffe_model);
            flags.push_back("--weights");
            flags.push_back(dcfg.caffe_weights);
            flags.push_back("--default_c");
            flags.push_back(dcfg.caffe_default_c);
            if (index >= 0) {
                flags.push_back("--gpu");
                flags.push_back(boost::lexical_cast<string, int>(index));
            }
            else {
                flags.push_back("--cpu");
                flags.push_back("-1");
            }
            flags.push_back("--solver_count");
            flags.push_back(boost::lexical_cast<string, int>(solver_count));

            std::chrono::time_point<clock_> beg_ = clock_::now();
            if (!caffep.init(flags, index)) {
                DEEP_LOG_ERROR("failed to start caffe !!!!!!!!!!!!!!!!!!!!!!! ");
            }
            double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
            DEEP_LOG_INFO("caffe initialization consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

            bool running = true;
            this->receive_while(running) (
                [=](const caffe_process_atom, uint64 datapointer) {
                DEEP_LOG_INFO("new request to caffe proc " + boost::lexical_cast<string>(index) + " to process data");
                caffe_batch_package *data = (caffe_batch_package*)datapointer;
                std::chrono::time_point<clock_> beg_ = clock_::now();
                caffep.predict_b(data->imgs, data->output);
                double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                DEEP_LOG_INFO("caffe predict " + boost::lexical_cast<string>(data->sources.size()) + " pics one time consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");
                for (int i = 0; i < data->sources.size(); ++i) {
                    //actor back = actor_cast<actor>(data->sources[i]->self);
                    actor& back = data->sources[i]->self;
                    data->sources[i]->time_consumed.predict_time = elapsed;
                    data->sources[i]->time_consumed.whole_time += elapsed;
                    //if (batch_size == 1) {
                    //    //data->sources[i]->output = data->output;
                    //}
                    //else {
                    //    FRCNN_API::Frcnn_wrapper::copySingleBlob(data->output, i, data->sources[i]->output);
                    //}

                    caffep.postprocess_b(data->imgs, i, data->output, data->sources[i]->results);
                    this->send(back, process_atom::value, (uint64)0);
                }
                delete data;
            },
            [=](tick_atom, size_t interval) {
                delayed_send(this, std::chrono::milliseconds{ interval },
                    tick_atom::value, interval);
            },
            [=](vector <std::pair<uint64, uint64> > a) {
            },
            [&](exit_msg& em) {
                if (em.reason) {
                    this->fail_state(std::move(em.reason));
                    running = false;
                }
            }
            );
        }

    private:
        caffe_process caffep;
        actor_config xcfg;
        int solver_count;
        int batch_size;
        int index;
        vector <std::pair<uint64, uint64> > batch_tasks;
    };

    class yolo_proc : public blocking_actor {
    public:
        yolo_proc(actor_config& cfg, int count, int i, int bz) : blocking_actor(cfg), xcfg(cfg), solver_count(count), index(i), batch_size(bz) {
        }

        void act() override {
            deepconfig& dcfg = (deepconfig&)xcfg.host->system().config();

            std::chrono::time_point<clock_> beg_ = clock_::now();
            if (!yolop.init(index, dcfg.yolo_cfgfile.c_str(), 
                dcfg.yolo_weightfile.c_str(), dcfg.yolo_namelist.c_str(), dcfg.yolo_labeldir.c_str(), 
                dcfg.yolo_thresh, dcfg.yolo_hier_thresh)) {
                DEEP_LOG_ERROR("failed to start yolo !!!!!!!!!!!!!!!!!!!!!!! ");
            }
            double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
            DEEP_LOG_INFO("yolo initialization consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

            bool running = true;
            this->receive_while(running) (
                [=](const yolo_process_atom, uint64 datapointer) {
                DEEP_LOG_INFO("new request to yolo proc " + boost::lexical_cast<string>(index) + " to process data");
                yolo_batch_package *data = (yolo_batch_package*)datapointer;
                std::chrono::time_point<clock_> beg_ = clock_::now();

                yolop.predict(data->input, data->output);
                yolop.postprocess(*data->input, data->output);

                double elapsed = std::chrono::duration_cast<mill_second_> (clock_::now() - beg_).count();
                DEEP_LOG_INFO("yolo predict " + boost::lexical_cast<string>(data->sources.size()) + " pics one time consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");
                for (int i = 0; i < data->sources.size(); ++i) {
                    actor& back = data->sources[i]->self;
                    data->sources[i]->output = data->output;
                    data->sources[i]->time_consumed.predict_time = elapsed;
                    data->sources[i]->time_consumed.whole_time += elapsed;

                    this->send(back, process_atom::value, (uint64)0);
                }
                delete data;
            },
                [=](tick_atom, size_t interval) {
                delayed_send(this, std::chrono::milliseconds{ interval },
                    tick_atom::value, interval);
            },
                [=](vector <std::pair<uint64, uint64> > a) {
            },
                [&](exit_msg& em) {
                if (em.reason) {
                    this->fail_state(std::move(em.reason));
                    running = false;
                }
            }
            );
        }

    private:
        yolo_process yolop;
        actor_config xcfg;
        int solver_count;
        int batch_size;
        int index;
        vector <std::pair<uint64, uint64> > batch_tasks;
    };

    class preprocess_proc : public blocking_actor {
    public:
        preprocess_proc(actor_config& cfg, actor cp, int bz) : blocking_actor(cfg), lib_p(cp), batch_size(bz), isprocessing(false){
        }

        void process(bool force) {
            if (getlibmode().yolo && (yolo_batch_tasks.size() >= batch_size || force)) {
                int size = batch_size < yolo_batch_tasks.size() ? batch_size : yolo_batch_tasks.size();
                if (size > 0) {
                    DEEP_LOG_INFO("yolo pre process received a new request!");
                    yolo_batch_package *p = new yolo_batch_package();
                    for (int i = 0; i < size; ++i) {
                        uint64 d = yolo_batch_tasks.front();
                        yolo_batch_tasks.pop_front();
                        yolo_data* data = (yolo_data*)d;
                        p->sources.push_back(data);

                        //todo , support 1 pic only, need to add the batch function
                        p->input = &data->input;

                        if (p->sources.size() > 0) {

                            this->send(lib_p, yolo_process_atom::value, (uint64)(uint64*)p);
                        }
                    }
                }
            }
            if (getlibmode().caffe && (caffe_batch_tasks.size() >= batch_size || force)) {
                int size = batch_size < caffe_batch_tasks.size() ? batch_size : caffe_batch_tasks.size();
                if (size > 0) {
                    DEEP_LOG_INFO("caffe pre process received a new request!");
                    caffe_batch_package *p = new caffe_batch_package();

                    for (int i = 0; i < size; ++i) {
                        uint64 d = caffe_batch_tasks.front();
                        caffe_batch_tasks.pop_front();
                        caffe_data* data = (caffe_data*)d;
                        p->sources.push_back(data);
                        p->imgs.push_back(data->cv_image);
                    }
                    if (p->sources.size() > 0) {
                        //if (batch_size == 1) {
                        //    // one pic no need to do the batch mode
                        //    if (!FRCNN_API::Frcnn_wrapper::preprocess(imgs[0], p->input)) {
                        //    }
                        //}
                        //else {
                        //    if (!FRCNN_API::Frcnn_wrapper::batch_preprocess(imgs, p->input)) {
                        //    }
                        //}

                        this->send(lib_p, caffe_process_atom::value, (uint64)(uint64*)p);
                    }
                }
            }
            if (getlibmode().tensorflow && (tf_batch_tasks.size() >= batch_size || force)) {
                batch_size = 1; // support batch 1 only
                int size = batch_size < tf_batch_tasks.size() ? batch_size : tf_batch_tasks.size();
                if (size > 0) {
                    DEEP_LOG_INFO("tensorflow pre process received a new request!");
                    tf_batch_package *p = new tf_batch_package();
                    for (int i = 0; i < size; ++i) {
                        uint64 d = tf_batch_tasks.front();
                        tf_batch_tasks.pop_front();
                        tf_data* data = (tf_data*)d;
                        p->sources.push_back(data);

                        //todo , support 1 pic only, need to add the batch function
                        p->imgs.push_back(data->cv_image);

                        if (p->sources.size() > 0) {
                            this->send(lib_p, tf_process_atom::value, (uint64)(uint64*)p);
                        }
                    }
                }
            }
        }

        void act() override {
            size_t interval = 30;

            this->delayed_send(this, std::chrono::milliseconds{ interval }, tick_atom::value, interval);

            bool running = true;
            this->receive_while(running) (
                [=](const lib_preprocess_atom, backend_mode_t t, actor& source, uint64 datapointer, actor& lib_p) {
                    isprocessing = true;
                    if (t == backend_mode_t::caffe) {
                        caffe_batch_tasks.push_back(datapointer);
                    }
                    if (t == backend_mode_t::yolo) {
                        yolo_batch_tasks.push_back(datapointer);
                    }
                    if (t == backend_mode_t::tensorflow) {
                        tf_batch_tasks.push_back(datapointer);
                    }
                    process(false);
                    delayed_send(this, std::chrono::milliseconds{ interval }, tick_atom::value, interval);
                    isprocessing = false;
                },
                [=](tick_atom, size_t interval) {
                    if (!isprocessing) {
                        process(true);
                        delayed_send(this, std::chrono::milliseconds{ interval }, tick_atom::value, interval);
                    }
                },
                [&](exit_msg& em) {
                    if (em.reason) {
                        this->fail_state(std::move(em.reason));
                        running = false;
                    }
                }
            );
        }

    private:
        int batch_size;
        bool isprocessing;
        actor lib_p;
        deque <uint64 > caffe_batch_tasks;
        deque <uint64 > yolo_batch_tasks;
        deque <uint64 > tf_batch_tasks;
    };

    struct deep_manager_state {
        int size;
        int active;
        vector<actor> caffe_frame_actors;
        vector<actor> yolo_frame_actors;
        vector<actor> tf_frame_actors;
        vector<actor> caffe_preprocess_actors;
        vector<actor> yolo_preprocess_actors;
        vector<actor> tf_preprocess_actors;
    };

    using deep_manager_s = caf::stateful_actor<deep_manager_state>;
    behavior deep_manager(deep_manager_s* self) {
        deepconfig& cfg = (deepconfig&)self->system().config();
        self->state.size = 0;
        bool isGpu = false;
        if (cfg.caffe_gpu.length() > 0 && getlibmode().caffe) {
            vector<string> s_gpus;
            boost::split(s_gpus, cfg.caffe_gpu, boost::is_any_of(","));
            int gpus = s_gpus.size();
            if (gpus > 0) {
                isGpu = true;
                for (auto i = 0; i < gpus; ++i) {
                    actor proc1;
                    proc1 = self->spawn<caffe_proc>(gpus, boost::lexical_cast<int>(s_gpus[i]), cfg.batch_size);
                    self->state.caffe_frame_actors.push_back(proc1);
                    auto proc2 = self->spawn<preprocess_proc>(proc1, cfg.batch_size);
                    self->state.caffe_preprocess_actors.push_back(proc2);
                    self->state.size += self->state.caffe_frame_actors.size();
                }
            }
        }
        if (cfg.yolo_gpu.length() > 0 && getlibmode().yolo) {
            vector<string> s_gpus;
            boost::split(s_gpus, cfg.yolo_gpu, boost::is_any_of(","));
            int gpus = s_gpus.size();
            if (gpus > 0) {
                isGpu = true;
                for (auto i = 0; i < gpus; ++i) {
                    actor proc1;
                    proc1 = self->spawn<yolo_proc>(gpus, boost::lexical_cast<int>(s_gpus[i]), cfg.batch_size);
                    self->state.yolo_frame_actors.push_back(proc1);
                    auto proc2 = self->spawn<preprocess_proc>(proc1, cfg.batch_size);
                    self->state.yolo_preprocess_actors.push_back(proc2);
                    self->state.size += self->state.yolo_frame_actors.size();
                }
            }
        }
        if (cfg.tf_gpu.length() > 0 && getlibmode().tensorflow) {
            vector<string> s_gpus;
            boost::split(s_gpus, cfg.tf_gpu, boost::is_any_of(","));
            int gpus = s_gpus.size();
            if (gpus > 0) {
                isGpu = true;
                for (auto i = 0; i < gpus; ++i) {
                    actor proc1;
                    proc1 = self->spawn<tf_proc>(gpus, boost::lexical_cast<int>(s_gpus[i]), cfg.batch_size);
                    self->state.tf_frame_actors.push_back(proc1);
                    auto proc2 = self->spawn<preprocess_proc>(proc1, cfg.batch_size);
                    self->state.tf_preprocess_actors.push_back(proc2);
                    self->state.size += self->state.tf_frame_actors.size();
                }
            }
        }
        
        if(!isGpu){
            int cpus = 1;
            if (cfg.cpu.length() > 0) {
                cpus = boost::lexical_cast<int>(cfg.cpu);
                cpus = cpus > 0 ? cpus : 1;
            }
            if (cpus > 0) {
                for (auto i = 0; i < cpus; ++i) {
                    actor proc1;
                    if (getlibmode().yolo) {
                        proc1 = self->spawn<yolo_proc>(1, -1, cfg.batch_size);
                        self->state.yolo_frame_actors.push_back(proc1);
                        auto proc2 = self->spawn<preprocess_proc>(proc1, cfg.batch_size);
                        self->state.yolo_preprocess_actors.push_back(proc2);
                        self->state.size += self->state.yolo_frame_actors.size();
                    }
                    if (getlibmode().caffe){
                        proc1 = self->spawn<caffe_proc>(1, -1, cfg.batch_size);
                        self->state.caffe_frame_actors.push_back(proc1);
                        auto proc2 = self->spawn<preprocess_proc>(proc1, cfg.batch_size);
                        self->state.caffe_preprocess_actors.push_back(proc2);
                        self->state.size += self->state.caffe_frame_actors.size();
                    }
                    if (getlibmode().tensorflow) {
                        proc1 = self->spawn<tf_proc>(1, -1, cfg.batch_size);
                        self->state.tf_frame_actors.push_back(proc1);
                        auto proc2 = self->spawn<preprocess_proc>(proc1, cfg.batch_size);
                        self->state.tf_preprocess_actors.push_back(proc2);
                        self->state.size += self->state.tf_frame_actors.size();
                    }
                }
            }
        }

        return{
            [=](const deep_manager_process_atom, backend_mode_t t, actor& source, uint64 datapointer) {
            DEEP_LOG_INFO("new request to deep manager to process data");

            if (self->state.size > 0) {
                if (t == backend_mode_t::caffe) {
                    int choosed = self->state.active % self->state.caffe_preprocess_actors.size();
                    self->send(self->state.caffe_preprocess_actors[choosed], lib_preprocess_atom::value, t,
                        source, datapointer, self->state.caffe_frame_actors[choosed]);
                }
                if (t == backend_mode_t::yolo) {
                    int choosed = self->state.active % self->state.yolo_preprocess_actors.size();
                    self->send(self->state.yolo_preprocess_actors[choosed], lib_preprocess_atom::value, t,
                        source, datapointer, self->state.yolo_frame_actors[choosed]);
                }
                if (t == backend_mode_t::tensorflow) {
                    int choosed = self->state.active % self->state.tf_preprocess_actors.size();
                    self->send(self->state.tf_preprocess_actors[choosed], lib_preprocess_atom::value, t,
                        source, datapointer, self->state.tf_frame_actors[choosed]);
                }
                ++self->state.active;
                if (self->state.active >= self->state.size) {
                    self->state.active = 0;
                }
            }
            else {
                DEEP_LOG_INFO("invalid request to caffe since there is no deep lib instance initialized.!");
                self->send(source, process_atom::value, datapointer);
            }
        }
        };
    }

    behavior server(broker* self, const bool& http_mode, const information info, actor cf_manager) {
        DEEP_LOG_INFO("server is running");

        self->set_down_handler([=,&info](const down_msg& dm) {
        });

        if(http_mode){
            return {
                    [=](const new_connection_msg& ncm) {
                        DEEP_LOG_INFO("server accepted new connection");

                        auto impl = self->fork(http_worker, ncm.handle, cf_manager);

                        self->monitor(impl);
                        self->link_to(impl);
                    }
            };
        }else{
            return {
                    [=](const new_connection_msg& msg) {
                        DEEP_LOG_INFO("server accepted new connection");

                        auto impl = self->fork(tcp_worker, msg.handle, cf_manager);

                        self->monitor(impl);
                        self->link_to(impl);
                    }
            };
        }
    }

#if defined(WIN32) || defined(_WIN32)
#else
    // signal handling for ctrl+c
    std::atomic<bool> shutdown_flag{false};
#endif

    void run_server(actor_system& system, const deepconfig& cfg) {
        DEEP_LOG_INFO("run in server mode");

        // get a scoped actor for the communication with our CURL actors
        scoped_actor self{system};

        // start the logger server
        actorlogger sys_logger(system, const_cast<deepconfig&>(cfg));
        sys_logger.start();
        actorlogger::set_current_actor_logger(&sys_logger);

#if defined(WIN32) || defined(_WIN32)
#else
        struct sigaction act;
        act.sa_handler = [](int) { shutdown_flag = true; };
        auto set_sighandler = [&] {
            if (sigaction(SIGINT, &act, nullptr) != 0) {
                DEEP_LOG_ERROR("fatal: cannot set signal handler");
                abort();
            }
        };
        set_sighandler();
#endif
        setlibmode(cfg.lib_mode);

        information info;
        auto cf_manager = self->spawn<monitored>(deep_manager);
        //auto cf_manager = self->spawn<deep_manager>();

        auto server_actor = system.middleman().spawn_server(server, cfg.port, cfg.http_mode, info, cf_manager);
        if (!server_actor) {
            DEEP_LOG_ERROR("failed to spawn server: " + system.render(server_actor.error()));
            return;
        }

#if defined(WIN32) || defined(_WIN32)
#else
        while (!shutdown_flag)
            std::this_thread::sleep_for(std::chrono::seconds(1));
        aout(self) << color::cyan << "received CTRL+C" << color::reset_endl;

        // await actors
        act.sa_handler = [](int) { abort(); };
        set_sighandler();
#endif

        aout(self) << color::cyan
                   << "await ... ; this may take a while "
                           "(press CTRL+C again to abort)"
                   << color::reset_endl;

        sys_logger.stop();
        self->await_all_other_actors_done();
    }

}

using namespace deep_server;

void caf_main(actor_system& system, const deepconfig& cfg) {
    run_server(system, cfg);
}

CAF_MAIN(io::middleman)

