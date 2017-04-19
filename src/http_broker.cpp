
#include "deep_server.hpp"

namespace deep_server{

    behavior http_worker(http_broker* self, connection_handle hdl, actor cf_manager) {
        // tell network backend to receive any number of bytes between 1 and 1024
        self->configure_read(hdl, receive_policy::at_most(1024));

        return {
                [=](const new_data_msg& msg) {
                        self->state.analyzer.parse(reinterpret_cast<const char*>(msg.buf.data()), msg.buf.size());
                        httpmessage& m = self->state.analyzer.parser->m;

                        if(self->state.analyzer.isDone()) { 
                            if (m.method == http_method::HTTP_POST ){
                                auto actor_processor = self->spawn<processor>("http_processor", self, msg.handle, cf_manager);
                                self->monitor(actor_processor);
                                self->link_to(actor_processor);
                            
                                self->send(actor_processor, input_atom::value, m.body);
                            }
                            else if (m.method == http_method::HTTP_GET) {
                                self->write(msg.handle, cstr_size(http_ok), http_ok);
                                self->flush(msg.handle);

                                DEEP_LOG_INFO("quit http broker");

                                self->quit();
                            }
                            else {
                                self->write(msg.handle, cstr_size(http_ok), http_ok);
                                self->flush(msg.handle);

                                DEEP_LOG_INFO("quit http broker");

                                self->quit();
                            }
                        }
                    //}
                },
                [=](connection_handle& handle, output_atom, std::string content) {
                    Json::Value root;
                    Json::StyledWriter writer;
                    root["data"] = content;
                    root["msg"] =  "ok";
                    root["status"] = 200;

                    std::string result = writer.write(root);

                    //self->write(handle, cstr_size(http_img_header), http_img_header);
                    self->write(handle, result.size(), result.c_str());
                    //self->write(handle, cstr_size(http_img_tail), http_img_tail);
                    self->flush(handle);

                    //aout(self) << result << endl;

                    DEEP_LOG_INFO("quit http broker");

                    self->quit();
                },
                [=](connection_handle& handle, output_atom, vector<unsigned char>& content) {
                    Json::Value root;
                    Json::StyledWriter writer;
                    root["data"] = base64_encode(content.data(), content.size());
                    root["msg"] = "ok";
                    root["status"] = 200;

                    std::string result = writer.write(root);

                    self->write(handle, result.size(), result.c_str());
                    self->flush(handle);

                    DEEP_LOG_INFO("quit http broker");

                    self->quit();
                },
                [=](connection_handle& handle, fault_atom, std::string reason) {
                    DEEP_LOG_INFO("fault happened !!!!  reason : " + reason);
                    Json::Value root;
                    Json::StyledWriter writer;
                    root["msg"] = "fail";
                    root["status"] = 500;

                    std::string result = writer.write(root);

                    self->write(handle, result.size(), result.c_str());
                    self->flush(handle);

                    self->quit();
                },
                [=](const connection_closed_msg&) {
                    //self->flush(hdl);
                    self->quit();
                }
        };
    }

}

