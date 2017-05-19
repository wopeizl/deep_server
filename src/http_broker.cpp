
#include "deep_server.hpp"
#include "caffe_process.hpp"
#include "yolo_process.hpp"

namespace deep_server{

    string generateHttpbody(string &body) {
        string head1 = "\r\nAccess-Control-Allow-Origin: *\r\n";
        string head2 = "Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept\r\n";
        string head3 = "Access-Control-Allow-Methods: POST,GET,OPTIONS\r\n";
        string content = head1 + head2 + head3 + string("content-length: ") + boost::lexical_cast<string>(body.size()) + "\r\n\r\n";
        return content + body;
    }

    behavior http_worker(http_broker* self, connection_handle hdl, actor cf_manager) {
        // tell network backend to receive any number of bytes between 1 and 1024
        self->configure_read(hdl, receive_policy::at_most(1024));
        
        return {
                [=](const new_data_msg& msg) {
                        self->state.analyzer.parse(reinterpret_cast<const char*>(msg.buf.data()), msg.buf.size());
                        httpmessage& m = self->state.analyzer.parser->m;

                        if(self->state.analyzer.isDone()) { 
                            if (m.method == http_method::HTTP_POST ){
                                actor actor_processor;
                                if (getlibmode() == yolo) {
                                    actor_processor = self->spawn<yolo_processor>("http_processor", self, msg.handle, cf_manager);
                                }
                                else {
                                    actor_processor = self->spawn<caffe_processor>("http_processor", self, msg.handle, cf_manager);
                                }

                                self->monitor(actor_processor);
                                self->link_to(actor_processor);
                            
                                self->send(actor_processor, input_atom::value, m.body);
                            }
                            else if (m.method == http_method::HTTP_GET) {
                                Json::Value root;
                                Json::StyledWriter writer;
                                root["msg"] = "not supported";
                                root["status"] = 200;

                                std::string result = writer.write(root);
                                result = generateHttpbody(result);

                                self->write(msg.handle, cstr_size(http_img_header), http_img_header);
                                self->write(msg.handle, result.size(), result.c_str());
                                self->flush(msg.handle);

                                DEEP_LOG_INFO("quit http broker");

                                self->quit();
                            }
                            else {
                                Json::Value root;
                                Json::StyledWriter writer;
                                root["msg"] = "not supported";
                                root["status"] = 200;

                                std::string result = writer.write(root);
                                result = generateHttpbody(result);

                                self->write(msg.handle, cstr_size(http_img_header), http_img_header);
                                self->write(msg.handle, result.size(), result.c_str());
                                self->write(msg.handle, cstr_size(http_img_tail), http_img_tail);
                                self->flush(msg.handle);

                                DEEP_LOG_INFO("quit http broker");

                                self->quit();
                            }
                        }
                    //}
                },
                [=](connection_handle& handle, output_atom, int resDataType, std::string callback, http_output out) {
                    Json::Value root;
                    Json::StyledWriter writer;
                    root["msg"] =  "ok";
                    root["status"] = 200;
                    root["callback"] = callback;
                    root["dataT"] = resDataType;

                    if (resDataType == FRCNN_RESULT) {
                        root["caffe_result"] = Json::Value(Json::arrayValue);
                        for (int i = 0; i < out.caffe_result.size(); ++i) {
                            Json::Value sub;
                            sub["x"] = out.caffe_result[i].x();
                            sub["y"] = out.caffe_result[i].y();
                            sub["w"] = out.caffe_result[i].w();
                            sub["h"] = out.caffe_result[i].h();
                            sub["class"] = out.caffe_result[i].class_();
                            sub["score"] = out.caffe_result[i].score();

                            root["caffe_result"][i] = sub;
                        }
                    }
                    else if (resDataType == YOLO_RESULT) {
                        root["yolo_result"] = Json::Value(Json::arrayValue);
                        for (int i = 0; i < out.yolo_result.size(); ++i) {
                            Json::Value sub;
                            sub["x"] = out.yolo_result[i].x();
                            sub["y"] = out.yolo_result[i].y();
                            sub["w"] = out.yolo_result[i].w();
                            sub["h"] = out.yolo_result[i].h();
                            sub["class"] = out.yolo_result[i].class_();
                            sub["score"] = out.yolo_result[i].score();

                            root["yolo_result"][i] = sub;
                        }
                    }
                    else {
                        root["data"] = out.sdata;
                    }

#ifdef OUPUT_INCLUDE_TIME_STAMP
                    root["whole_time"] = out.ts.whole_time;
                    root["preprocess_time"] = out.ts.preprocess_time;
                    root["predict_time"] = out.ts.predict_time;
                    root["prepare_time"] = out.ts.prepare_time;
                    root["postprocess_time"] = out.ts.postprocess_time;
                    root["jsonparse_time"] = out.ts.jsonparse_time;
                    root["decode_time"] = out.ts.decode_time;
                    root["cvreadimage_time"] = out.ts.cvreadimage_time;
                    root["writeresult_time"] = out.ts.writeresult_time;

                    root["width"] = out.info.width();
                    root["height"] = out.info.height();
                    root["channel"] = out.info.channel();
#endif

                    std::string result = writer.write(root);
                    result = generateHttpbody(result);

                    self->write(handle, cstr_size(http_img_header), http_img_header);
                    self->write(handle, result.size(), result.c_str());
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
                    result = generateHttpbody(result);

                    self->write(handle, cstr_size(http_img_header), http_img_header);
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
                    result = generateHttpbody(result);

                    self->write(handle, cstr_size(http_img_header), http_img_header);
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

