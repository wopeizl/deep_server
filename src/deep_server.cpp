
#include "color.h"
#include "caffe_process.hpp"
#include "yolo_process.hpp"

namespace deep_server{

    static lib_mode_t g_lib_mode;

    void setlibmode(string mode) {
        if (mode == "caffe")
            g_lib_mode = caffe;
        else if (mode == "yolo")
            g_lib_mode = yolo;
        else
            g_lib_mode = caffe;
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
            if (batch_tasks.size() > 0) {
                if (batch_tasks.size() >= batch_size || force) {
                    DEEP_LOG_INFO("pre process received a new request!");

                    int size = batch_size < batch_tasks.size() ? batch_size : batch_tasks.size();
                    if (getlibmode() == yolo) {
                        yolo_batch_package *p = new yolo_batch_package();
                        for (int i = 0; i < size; ++i) {
                            uint64 d = batch_tasks.front();
                            batch_tasks.pop_front();
                            yolo_data* data = (yolo_data*)d;
                            p->sources.push_back(data);

                            //todo , support 1 pic only, need to add the batch function
                            p->input = &data->input;

                            if (p->sources.size() > 0) {

                                this->send(lib_p, yolo_process_atom::value, (uint64)(uint64*)p);
                            }
                        }
                    }
                    else {
                        caffe_batch_package *p = new caffe_batch_package();

                        for (int i = 0; i < size; ++i) {
                            uint64 d = batch_tasks.front();
                            batch_tasks.pop_front();
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
            }
        }

        void act() override {
            size_t interval = 30;

            this->delayed_send(this, std::chrono::milliseconds{ interval }, tick_atom::value, interval);

            bool running = true;
            this->receive_while(running) (
                [=](const lib_preprocess_atom, actor& source, uint64 datapointer, actor& lib_p) {
                    isprocessing = true;
                    batch_tasks.push_back(datapointer);
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
        deque <uint64 > batch_tasks;
    };

    struct deep_manager_state {
        int size;
        int active;
        vector<actor> frame_actors;
        vector<actor> preprocess_actors;
    };

    using deep_manager_s = caf::stateful_actor<deep_manager_state>;
    behavior deep_manager(deep_manager_s* self) {
        deepconfig& cfg = (deepconfig&)self->system().config();
        if (cfg.gpu.length() > 0) {
            vector<string> s_gpus;
            boost::split(s_gpus, cfg.gpu, boost::is_any_of(","));
            int gpus = s_gpus.size();
            if (gpus > 0) {
                for (auto i = 0; i < gpus; ++i) {
                    actor proc1;
                    if (getlibmode() == yolo) {
                        proc1 = self->spawn<yolo_proc>(gpus, boost::lexical_cast<int>(s_gpus[i]), cfg.batch_size);
                    }
                    else {
                        proc1 = self->spawn<caffe_proc>(gpus, boost::lexical_cast<int>(s_gpus[i]), cfg.batch_size);
                    }
                    self->state.frame_actors.push_back(proc1);

                    auto proc2 = self->spawn<preprocess_proc>(proc1, cfg.batch_size);
                    self->state.preprocess_actors.push_back(proc2);

                    self->state.size = self->state.frame_actors.size();
                }
            }
        }
        else {
            int cpus = 1;
            if (cfg.cpu.length() > 0) {
                cpus = boost::lexical_cast<int>(cfg.cpu);
                cpus = cpus > 0 ? cpus : 1;
            }
            if (cpus > 0) {
                for (auto i = 0; i < cpus; ++i) {
                    actor proc1;
                    if (getlibmode() == yolo) {
                        proc1 = self->spawn<yolo_proc>(1, -1, cfg.batch_size);
                    }
                    else {
                        proc1 = self->spawn<caffe_proc>(1, -1, cfg.batch_size);
                    }
                    self->state.frame_actors.push_back(proc1);

                    auto proc2 = self->spawn<preprocess_proc>(proc1, cfg.batch_size);
                    self->state.preprocess_actors.push_back(proc2);

                    self->state.size = self->state.frame_actors.size();
                }
            }
        }

        return{
            [=](const deep_manager_process_atom, actor& source, uint64 datapointer) {
            DEEP_LOG_INFO("new request to deep manager to process data");

            if (self->state.size > 0) {
                self->send(self->state.preprocess_actors[self->state.active], lib_preprocess_atom::value, 
                    source, datapointer, self->state.frame_actors[self->state.active]);

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

