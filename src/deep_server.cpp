
#define NO_STRICT 
#include "color.h"
#include "cv_process.hpp"
#include "caffe_process.hpp"
#include "deep_server.hpp"

namespace deep_server{

    // utility function to print an exit message with custom name
    void print_on_exit(const actor& hdl, const std::string& name) {
        hdl->attach_functor([=](const caf::error& reason) {
            DEEP_LOG_INFO(name + " exited: " + to_string(reason));
        });
    }

    struct caffe_data {
        cv::Mat cv_image;
        std::vector<caffe::Frcnn::BBox<float>> results;
        connection_handle handle;
    };

    processor::processor(actor_config& cfg,
        std::string  n, actor s, connection_handle handle, actor cm)
        : event_based_actor(cfg),
        name_(std::move(n)), bk(s), handle_(handle), cf_manager(cm) {

        //set_default_handler(skip);

        uploader_.assign(
            [=](uploader_atom, uint64 datapointer) {

            become(processor_);
            send(cf_manager, caffe_manager_process_atom::value, this, datapointer);
        }
        );

        processor_.assign(
            [=](process_atom, uint64 datapointer) {

            DEEP_LOG_INFO(name_ + " starts to process_atom.");

            caffe_data* data = (caffe_data*)datapointer;
            if (!cvprocess::process_caffe_result(data->results, data->cv_image)) {
                fault("fail to cvprocess::process_caffe_result ! ");
                return;
            }

            vector<unsigned char> odata;
            if (!cvprocess::writeImage(data->cv_image, odata)) {
                fault("fail to cvprocess::process_caffe_result ! ");
                return;
            }

            connection_handle handle = data->handle;

            delete data;

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
            if (!reader.parse(data, value) || value.isNull() || !value.isObject()) {
                fault("解析请求失败，不是有效的json");
                return;
            }

            std::string out = value.get("data", "").asString();
            vector<unsigned char> idata;
            idata = base64_decode(out);
            if (out.length() == 0 || idata.size() == 0) {
                fault("data属性无效，请检查.");
                return;
            }

            std::string method = value.get("method", "").asString();
            if (method == "flip") {
                std::vector<unsigned char> odata;
                if (!cvprocess::flip(idata, odata)) {
                    fault("图片" + method + "操作失败，请检查是否为有效图片！");
                    return;
                }

                become(downloader_);
                send(this, downloader_atom::value, handle_, odata);
            }
            else if (method == "caffe") {
                caffe_data* data = new caffe_data();

                if (!cvprocess::readImage(idata, data->cv_image)) {
                    fault("图片操作失败，请检查是否为有效图片！");
                    return;
                }

                data->handle = handle_;

                become(uploader_);
                send(this, uploader_atom::value, (uint64)(uint64*)data);
            }
        }
        );
    }

    class caffe_proc : public blocking_actor{
    public:
        caffe_proc(actor_config& cfg) : blocking_actor(cfg) {
            vector<string> flags;
            deepconfig& dcfg = (deepconfig&)cfg.host->system().config();
            flags.push_back("");
            flags.push_back("--model");
            //flags.push_back("data\test.prototxt");
            flags.push_back(dcfg.model);
            flags.push_back("--weights");
            //flags.push_back("data\VGG16_faster_rcnn_final.caffemodel");
            flags.push_back(dcfg.weights);
            flags.push_back("--default_c");
            //flags.push_back("data\voc_config.json");
            flags.push_back(dcfg.default_c);

            if (!caffep.init(flags)) {
                DEEP_LOG_ERROR("failed to start caffe !!!!!!!!!!!!!!!!!!!!!!! ");
            }
        }

        void act() override {
            bool running = true;
            this->receive_while(running) (
                [=](const caffe_process_atom, actor& source, uint64 datapointer) {
                DEEP_LOG_INFO("new request to caffe to process data");

                caffe_data* data = (caffe_data*)datapointer;

                caffep.predict(data->cv_image, data->results);

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
    };

    struct caffe_manager_state {
        int size;
        int active;
        vector<actor> caffe_actors;
    };

    using caffe_manager_s = caf::stateful_actor<caffe_manager_state>;
    behavior caffe_manager(caffe_manager_s* self) {
        
        auto proc = self->spawn<caffe_proc>();
        self->state.caffe_actors.push_back(proc);
        self->state.size = self->state.caffe_actors.size();

        return{
            [=](const caffe_manager_process_atom, actor& source, uint64 datapointer) {
            DEEP_LOG_INFO("new request to caffe to process data");

            self->send(self->state.caffe_actors[self->state.active], caffe_process_atom::value, source, datapointer);

            ++self->state.active;
            if (self->state.active >= self->state.size) {
                self->state.active = 0;
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

