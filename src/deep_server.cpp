
#include "color.h"
#include "deep_server.hpp"

namespace deep_server{

    // utility function to print an exit message with custom name
    void print_on_exit(const actor& hdl, const std::string& name) {
        hdl->attach_functor([=](const caf::error& reason) {
            DEEP_LOG_INFO(name + " exited: " + to_string(reason));
        });
    }

    processor::processor(actor_config& cfg,
        std::string  n, actor s, connection_handle handle, actor cm)
        : event_based_actor(cfg),
        name_(std::move(n)), bk(s), handle_(handle), cf_manager(cm){

        pcaffe_data.reset(new caffe_data());
        pcaffe_data->time_consumed = 0.f;

        //set_default_handler(skip);

        prepare_.assign(
            [=](prepare_atom, uint64 datapointer) {

            std::chrono::time_point<clock_> beg_ = clock_::now();
            if (!FRCNN_API::Frcnn_wrapper::prepare(pcaffe_data->cv_image, pcaffe_data->out_image)) {
                fault("Fail to prepare image£¡");
                return;
            }
            double elapsed = std::chrono::duration_cast<micro_second_> (clock_::now() - beg_).count();
            pcaffe_data->time_consumed += elapsed;
            DEEP_LOG_INFO("prepare caffe consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

            become(preprocess_);
            send(this, preprocess_atom::value, datapointer);
        }
        );

        preprocess_.assign(
            [=](preprocess_atom, uint64 datapointer) {

            std::chrono::time_point<clock_> beg_ = clock_::now();
            if (!FRCNN_API::Frcnn_wrapper::preprocess(pcaffe_data->out_image, pcaffe_data->input)) {
                fault("Fail to preprocess image£¡");
                return;
            }
            double elapsed = std::chrono::duration_cast<micro_second_> (clock_::now() - beg_).count();
            pcaffe_data->time_consumed += elapsed;
            DEEP_LOG_INFO("preprocess caffe consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

            become(processor_);
            send(cf_manager, caffe_manager_process_atom::value, this, datapointer);
        }
        );

        processor_.assign(
            [=](process_atom, uint64 datapointer) {

            DEEP_LOG_INFO(name_ + " starts to process_atom.");

            caffe_data* data = (caffe_data*)datapointer;

            std::chrono::time_point<clock_> beg_ = clock_::now();
            if (!FRCNN_API::Frcnn_wrapper::postprocess(data->cv_image, data->output, data->results)) {
                fault("fail to Frcnn_wrapper::postprocess ! ");
                return;
            }
            double elapsed = std::chrono::duration_cast<micro_second_> (clock_::now() - beg_).count();
            pcaffe_data->time_consumed += elapsed;
            DEEP_LOG_INFO("postprocess caffe consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

            beg_ = clock_::now();
            if (!cvprocess::process_caffe_result(data->results, data->cv_image)) {
                fault("fail to cvprocess::process_caffe_result ! ");
                return;
            }
            elapsed = std::chrono::duration_cast<micro_second_> (clock_::now() - beg_).count();
            pcaffe_data->time_consumed += elapsed;
            DEEP_LOG_INFO("process caffe result consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

            beg_ = clock_::now();
            vector<unsigned char> odata;
            if (!cvprocess::writeImage(data->cv_image, odata)) {
                fault("fail to cvprocess::process_caffe_result ! ");
                return;
            }
            elapsed = std::chrono::duration_cast<micro_second_> (clock_::now() - beg_).count();
            pcaffe_data->time_consumed += elapsed;
            DEEP_LOG_INFO("write result consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

            connection_handle handle = data->handle;

            become(downloader_);
            send(this, downloader_atom::value, handle, odata);
            }
        );
        downloader_.assign(
            [=](downloader_atom, connection_handle handle, vector<unsigned char>& odata) {

            DEEP_LOG_INFO(name_ + " starts to downloader_atom.");

            send(bk, handle, output_atom::value, base64_encode(odata.data(), odata.size()));
            //send(bk, handle_, output_atom::value, odata);
        }
        );
    }

    behavior processor::make_behavior()  {
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
            double elapsed = std::chrono::duration_cast<micro_second_> (clock_::now() - beg_).count();
            pcaffe_data->time_consumed += elapsed;
            DEEP_LOG_INFO("parse json consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

            std::string out = value.get("data", "").asString();
            vector<unsigned char> idata;
            beg_ = clock_::now();
            idata = base64_decode(out);
            if (out.length() == 0 || idata.size() == 0) {
                fault("invalid json property : data.");
                return;
            }
            elapsed = std::chrono::duration_cast<micro_second_> (clock_::now() - beg_).count();
            pcaffe_data->time_consumed += elapsed;
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
            else if (method == "caffe") {

                std::chrono::time_point<clock_> beg_ = clock_::now();
                if (!cvprocess::readImage(idata, pcaffe_data->cv_image)) {
                    fault("Fail to read image£¡");
                    return;
                }
                double elapsed = std::chrono::duration_cast<micro_second_> (clock_::now() - beg_).count();
                pcaffe_data->time_consumed += elapsed;
                DEEP_LOG_INFO("cv read image consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                pcaffe_data->handle = handle_;

                become(prepare_);
                send(this, prepare_atom::value, (uint64)(uint64*)pcaffe_data.get());
            }
            else {
                fault("invalid method : £¡" + method);
                return;
            }
        }
        );
    }

    class caffe_proc : public blocking_actor{
    public:
        caffe_proc(actor_config& cfg, bool isgpu_, int i) : blocking_actor(cfg), xcfg(cfg), isgpu(isgpu_), index(i) {
        }

        void act() override {
            vector<string> flags;
            deepconfig& dcfg = (deepconfig&)xcfg.host->system().config();
            flags.push_back("");
            flags.push_back("--model");
            flags.push_back(dcfg.model);
            flags.push_back("--weights");
            flags.push_back(dcfg.weights);
            flags.push_back("--default_c");
            flags.push_back(dcfg.default_c);
            if (isgpu) {
                flags.push_back("--gpu");
                flags.push_back(boost::lexical_cast<string, int>(index));
            }
            else {
                flags.push_back("--cpu");
                flags.push_back("-1");
            }

            std::chrono::time_point<clock_> beg_ = clock_::now();
            if (!caffep.init(flags)) {
                DEEP_LOG_ERROR("failed to start caffe !!!!!!!!!!!!!!!!!!!!!!! ");
            }
            double elapsed = std::chrono::duration_cast<micro_second_> (clock_::now() - beg_).count();
            DEEP_LOG_INFO("caffe initialization consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

            bool running = true;
            this->receive_while(running) (
                [=](const caffe_process_atom, actor& source, uint64 datapointer) {
                DEEP_LOG_INFO("new request to caffe proc to process data");

                caffe_data* data = (caffe_data*)datapointer;
                std::chrono::time_point<clock_> beg_ = clock_::now();
                caffep.predict(data->input, data->output);
                double elapsed = std::chrono::duration_cast<micro_second_> (clock_::now() - beg_).count();
                data->time_consumed += elapsed;
                DEEP_LOG_INFO("caffe predict consume time : " + boost::lexical_cast<string>(elapsed) + "ms!");

                this->send(source, process_atom::value, datapointer);
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
        bool isgpu;
        int index;
    };

    struct caffe_manager_state {
        int size;
        int active;
        vector<actor> caffe_actors;
    };

    //class caffe_manager : public blocking_actor {
    //public:
    //    caffe_manager(actor_config& cfg) : blocking_actor(cfg) {
    //        *xcfg = cfg;
    //    }

    //    void act() override {
    //        deepconfig& dcfg = (deepconfig&)xcfg->host->system().config();
    //        int gpus = boost::lexical_cast<int>(dcfg.gpu);
    //        for (auto i = 0; i < gpus; ++i) {
    //            auto proc = this->spawn<caffe_proc>();
    //            this->state.caffe_actors.push_back(proc);
    //            this->state.size = this->state.caffe_actors.size();
    //        }
    //        bool running = true;
    //        this->receive_while(running) (
    //            [=](const caffe_manager_process_atom, actor& source, uint64 datapointer) {
    //                DEEP_LOG_INFO("new request to caffe to process data");

    //                if (this->state.size > 0) {
    //                    this->send(this->state.caffe_actors[this->state.active], caffe_process_atom::value, source, datapointer);

    //                    ++this->state.active;
    //                    if (this->state.active >= this->state.size) {
    //                        this->state.active = 0;
    //                    }
    //                }
    //                else {
    //                    DEEP_LOG_INFO("invalid request to caffe since there is no caffe instance initialized.!");
    //                    this->send(source, process_atom::value, datapointer);
    //                }
    //            },
    //            [&](exit_msg& em) {
    //            if (em.reason) {
    //                this->fail_state(std::move(em.reason));
    //                running = false;
    //            }
    //        }
    //        );
    //    }

    //private:
    //    actor_config* xcfg;
    //    caffe_manager_state state;
    //};

    using caffe_manager_s = caf::stateful_actor<caffe_manager_state>;
    behavior caffe_manager(caffe_manager_s* self) {
        deepconfig& cfg = (deepconfig&)self->system().config();
        if (cfg.gpu.length() > 0) {
            int gpus = boost::lexical_cast<int>(cfg.gpu);
            if (gpus > 0) {
                for (auto i = 0; i < gpus; ++i) {
                    auto proc = self->spawn<caffe_proc>(true, i);
                    self->state.caffe_actors.push_back(proc);
                    self->state.size = self->state.caffe_actors.size();
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
                    auto proc = self->spawn<caffe_proc>(false, i);
                    self->state.caffe_actors.push_back(proc);
                    self->state.size = self->state.caffe_actors.size();
                }
            }
        }

        return{
            [=](const caffe_manager_process_atom, actor& source, uint64 datapointer) {
            DEEP_LOG_INFO("new request to caffemanager to process data");

            if (self->state.size > 0) {
                self->send(self->state.caffe_actors[self->state.active], caffe_process_atom::value, source, datapointer);

                ++self->state.active;
                if (self->state.active >= self->state.size) {
                    self->state.active = 0;
                }
            }
            else {
                DEEP_LOG_INFO("invalid request to caffe since there is no caffe instance initialized.!");
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

                        auto impl = self->fork(tcp_broker, msg.handle, cf_manager);

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
        information info;
        auto cf_manager = self->spawn<monitored>(caffe_manager);
        //auto cf_manager = self->spawn<caffe_manager>();

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

