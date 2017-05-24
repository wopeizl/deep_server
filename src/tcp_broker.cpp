
#include "deep_server.hpp"
#include "caffe_process.hpp"
#include "yolo_process.hpp"

namespace deep_server {

    // utility function for sending an integer type
    template <class T>
    void write_int(broker* self, connection_handle hdl, T value) {
        using unsigned_type = typename std::make_unsigned<T>::type;
        auto cpy = static_cast<T>(htonl(static_cast<unsigned_type>(value)));
        self->write(hdl, sizeof(T), &cpy);
        self->flush(hdl);
    }

    void write_int(broker* self, connection_handle hdl, uint64_t value) {
        // write two uint32 values instead (htonl does not work for 64bit integers)
        write_int(self, hdl, static_cast<uint32_t>(value));
        write_int(self, hdl, static_cast<uint32_t>(value >> sizeof(uint32_t)));
    }

    // utility function for reading an ingeger from incoming data
    template <class T>
    void read_int(const void* data, T& storage) {
        using unsigned_type = typename std::make_unsigned<T>::type;
        memcpy(&storage, data, sizeof(T));
        storage = static_cast<T>(ntohl(static_cast<unsigned_type>(storage)));
    }

    void read_int(const void* data, uint64_t& storage) {
        uint32_t first;
        uint32_t second;
        read_int(data, first);
        read_int(reinterpret_cast<const char*>(data) + sizeof(uint32_t), second);
        storage = first | (static_cast<uint64_t>(second) << sizeof(uint32_t));
    }

    //// implemenation of our broker
    void tcp_worker(broker* self, connection_handle hdl, actor cf_manager) {
        self->set_down_handler([=](down_msg& dm) {
            self->quit();
        });

        auto write = [=](const deep_server::output_m& p) {
            string buf = p.SerializeAsString();
            auto s = htonl(static_cast<uint32_t>(buf.size()));
            self->write(hdl, sizeof(uint32_t), &s);
            self->write(hdl, buf.size(), buf.data());
            self->flush(hdl);
        };
        message_handler default_bhvr = {
            [=](const connection_closed_msg& msg) {
            DEEP_LOG_INFO("connection closed");

            if (msg.handle == hdl) {
                self->quit();
            }
            return;
        },
        [=](connection_handle& handle, fault_atom, std::string reason) {
            DEEP_LOG_INFO("fault happened !!!! reason : " + reason);

            self->quit();
        },
        [=](connection_handle& handle, output_atom, int resDataType, std::string callback, tcp_output out) {
            deep_server::output_m op;
            op.set_status(deep_server::Status::OK);
            op.set_msg("ok");
            op.set_callback(callback);
            op.set_datat((deep_server::dataType)resDataType);
            if(resDataType == FRCNN_RESULT){
                for (int i = 0; i < out.caffe_result.size(); ++i) {
                    Caffe_result* cr = op.add_caffe_result();
                    cr->CopyFrom(out.caffe_result[i]);
                }
            }
            else if(resDataType == YOLO_RESULT){
                for (int i = 0; i < out.yolo_result.size(); ++i) {
                    Yolo_result* cr = op.add_yolo_result();
                    cr->CopyFrom(out.yolo_result[i]);
                }
            }
            else {
                op.set_imgdata(out.bdata.data(), out.bdata.size());
            }
#ifdef OUPUT_INCLUDE_TIME_STAMP
            consume_time t;
            t.set_cvreadimage_time(out.ts.cvreadimage_time);
            t.set_predict_time(out.ts.predict_time);
            t.set_decode_time(out.ts.decode_time);
            t.set_postprocess_time(out.ts.postprocess_time);
            t.set_prepare_time(out.ts.prepare_time);
            t.set_preprocess_time(out.ts.preprocess_time);
            t.set_whole_time(out.ts.whole_time);
            t.set_writeresult_time(out.ts.writeresult_time);
            op.mutable_time()->CopyFrom(t);

            Img_info info;
            info.set_channel(out.info.channel());
            info.set_width(out.info.width());
            info.set_height(out.info.height());
            op.mutable_info()->CopyFrom(info);
#endif
            write(op);
        }
        };
        auto await_protobuf_data = message_handler{
            [=](const new_data_msg& msg) {
            deep_server::input_m p;
            p.ParseFromArray(msg.buf.data(), static_cast<int>(msg.buf.size()));

            if (p.has_method() && p.has_imgdata()) {
                if (p.method() == deep_server::input_m::CV_FLIP) {
                    std::vector<unsigned char> odata;
                    vector<unsigned char> idata(p.imgdata().size());
                    std::memcpy(idata.data(), (char*)p.imgdata().data(), p.imgdata().size());
                    deep_server::output_m op;
                    if (!cvprocess::flip(idata, odata)) {
                        DEEP_LOG_ERROR("Fail to process pic by " + boost::lexical_cast<string>(p.method()) + ", please check£¡");
                        op.set_status(deep_server::Status::SEVER_FAIL);
                    }
                    else {
                        op.set_status(deep_server::Status::OK);
                        op.set_datat(deep_server::CV_POST_PNG);
                        op.set_imgdata(odata.data(), odata.size());
                        op.set_callback(p.callback());
                    }
                    int a = odata.size();
                    write(op);
                }
                else if ((p.method() == deep_server::input_m::CAFFE && getlibmode() == caffe)
                    ||(p.method() == deep_server::input_m::YOLO && getlibmode() == yolo)) {
                    actor actor_processor;
                    vector<unsigned char> idata(p.imgdata().size());
                    std::memcpy(idata.data(), (char*)p.imgdata().data(), p.imgdata().size());
                    if (getlibmode() == yolo) {
                        actor_processor = self->spawn<yolo_processor>("tcp_processor", self, msg.handle, cf_manager);
                    }
                    else {
                        actor_processor = self->spawn<caffe_processor>("tcp_processor", self, msg.handle, cf_manager);
                    }
                    self->monitor(actor_processor);
                    self->link_to(actor_processor);

                    self->send(actor_processor, input_atom::value, (int)p.datat(), (int)p.res_datat(), idata);
                }
                else {
                    deep_server::output_m op;
                    op.set_status(deep_server::Status::WRONG_METHOD);
                    write(op);
                }
            }else{
                deep_server::output_m op;
                op.set_status(deep_server::Status::INVALID);
                write(op);
            }

            // receive next length prefix
            self->configure_read(hdl, receive_policy::exactly(sizeof(uint32_t)));
            self->unbecome();
        }
        }.or_else(default_bhvr);
        auto await_length_prefix = message_handler{
            [=](const new_data_msg& msg) {
            int num_bytes;
            memcpy(&num_bytes, msg.buf.data(), sizeof(uint32_t));
            num_bytes = ntohl(num_bytes);
            if (num_bytes > (32 * 1024 * 1024)) {
                DEEP_LOG_ERROR("The request pic size is too large > 32 M ");

                deep_server::output_m p;
                p.set_status(deep_server::Status::TOO_BIG_DATA);
                p.set_msg("too large pic");
                write(p);

                self->quit();
            }
            else {
                // receive protobuf data
                auto nb = static_cast<size_t>(num_bytes);
                self->configure_read(hdl, receive_policy::exactly(nb));
                self->become(keep_behavior, await_protobuf_data);
            }
        }
        }.or_else(default_bhvr);
        // initial setup
        self->configure_read(hdl, receive_policy::exactly(sizeof(uint32_t)));
        self->become(await_length_prefix);
    }
}