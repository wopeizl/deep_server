
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

    // implemenation of our broker
    void tcp_broker(broker* self, connection_handle hdl, actor cf_manager) {
        self->set_down_handler([=](down_msg& dm) {
            self->quit(exit_reason::remote_link_unreachable);
            return;
        });

        auto write = [=](const org::libcppa::output_m& p) {
            string buf = p.SerializeAsString();
            auto s = htonl(static_cast<uint32_t>(buf.size()));
            self->write(hdl, sizeof(uint32_t), &s);
            self->write(hdl, buf.size(), buf.data());
            self->flush(hdl);
        };
        message_handler default_bhvr = {
            [=](const connection_closed_msg&) {
            DEEP_LOG_INFO("connection closed");
            //self->send_exit(buddy, exit_reason::remote_link_unreachable);
            self->quit(exit_reason::remote_link_unreachable);
            return;
        },
            [=](connection_handle& handle, fault_atom, std::string reason) {
            DEEP_LOG_INFO("fault happened !!!! reason : " + reason);

            self->quit();
            return;
        },
        //    [=](connection_handle& handle, output_atom, org::libcppa::output_m& data) {
        //    string buf = data.SerializeAsString();
        //    auto s = htonl(static_cast<uint32_t>(buf.size()));
        //    self->write(hdl, sizeof(uint32_t), &s);
        //    self->write(hdl, buf.size(), buf.data());
        //    self->flush(hdl);

        //    DEEP_LOG_INFO("quit tcp broker");
        //    self->quit();
        //},
            [=](connection_handle& handle, output_atom, vector<unsigned char>& odata) {
            org::libcppa::output_m p;
            p.set_imgdata(odata.data(), odata.size());
            write(p);
        }
        };
        auto await_protobuf_data = message_handler{
            [=](const new_data_msg& msg) {
            org::libcppa::input_m p;
            p.ParseFromArray(msg.buf.data(), static_cast<int>(msg.buf.size()));

            if (p.has_method() && p.has_imgdata()) {
                actor actor_processor;

                if (p.method() == org::libcppa::input_m::CV_FLIP) {
                    std::vector<unsigned char> odata;
                    vector<unsigned char> idata(p.imgdata().size());
                    std::memcpy(idata.data(), (char*)p.imgdata().data(), p.imgdata().size());
                    org::libcppa::output_m op;
                    if (!cvprocess::flip(idata, odata)) {
                        DEEP_LOG_ERROR("Fail to process pic by " + boost::lexical_cast<string>(p.method()) + ", please check£¡");
                        op.set_status(org::libcppa::Status::SEVER_FAIL);
                    }
                    else {
                        op.set_imgdata(odata.data(), odata.size());
                    }

                    write(op);
                }
                else if ((p.method() == org::libcppa::input_m::CAFFE && getlibmode() == caffe)
                    ||(p.method() == org::libcppa::input_m::YOLO && getlibmode() == yolo)) {
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

                    self->send(actor_processor, input_atom::value, (int)p.datat(), idata);
                }
                else {
                    org::libcppa::output_m op;
                    op.set_status(org::libcppa::Status::WRONG_METHOD);
                    write(op);
                }
            }else{
                org::libcppa::output_m op;
                op.set_status(org::libcppa::Status::INVALID);
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
            if (num_bytes > (100 * 1024 * 1024)) {
                DEEP_LOG_ERROR("someone is trying something nasty");

                org::libcppa::output_m p;
                p.set_status(org::libcppa::Status::TOO_BIG_DATA);
                p.set_msg("too large pic");
                write(p);

                self->quit(exit_reason::user_shutdown);
                return;
            }
            // receive protobuf data
            auto nb = static_cast<size_t>(num_bytes);
            self->configure_read(hdl, receive_policy::exactly(nb));
            self->become(keep_behavior, await_protobuf_data);
        }
        }.or_else(default_bhvr);
        // initial setup
        self->configure_read(hdl, receive_policy::exactly(sizeof(uint32_t)));
        self->become(await_length_prefix);
    }

}