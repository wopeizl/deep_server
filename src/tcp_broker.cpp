
#include "deep_server.hpp"

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
    behavior tcp_broker(broker* self, connection_handle hdl, actor cf_manager) {
        self->set_down_handler([=](down_msg& dm) {
        });

        self->configure_read(hdl, receive_policy::exactly(sizeof(uint64_t) +
            sizeof(int32_t)));

        return{
            [=](const connection_closed_msg& msg) {
            if (msg.handle == hdl) {
                DEEP_LOG_INFO("connection closed");
                //self->send_exit(buddy, exit_reason::remote_link_unreachable);
                self->quit(exit_reason::remote_link_unreachable);
            }
        },
            [=](const new_data_msg& msg) {
            uint64_t atm_val;
            read_int(msg.buf.data(), atm_val);
            auto atm = static_cast<atom_value>(atm_val);
            int32_t ival;
            read_int(msg.buf.data() + sizeof(uint64_t), ival);
            DEEP_LOG_INFO("received {" << to_string(atm) << ", " << ival << "}");

            auto actor_processor = self->spawn<processor>("tcp_processor", self, msg.handle, cf_manager);
            self->monitor(actor_processor);
            self->link_to(actor_processor);
            vector<unsigned char> idata;
            self->send(actor_processor, input_atom::value, idata);
        },
        [=](connection_handle& handle, fault_atom, std::string reason) {
            DEEP_LOG_INFO("fault happened !!!! reason : " + reason);

            self->quit();
        }
        };
    }

}