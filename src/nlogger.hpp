#pragma once
#include "deep_server.hpp"

namespace deep_server {

    class nlogger {
    public:
        enum class level : int {
            quiet = 0,
            error = 1,
            info = 2,
            verbose = 3,
            massive = 4
        };

        static bool init(int lvl_cons, int lvl_file, const std::string& logfile);

        static nlogger& instance();

        template <class T>
        void log(level lvl, const T& x) {
            if (lvl <= level_console_) {
                *console_ << x;
            }
            if (lvl <= level_file_) {
                file_ << x;
            }
        }

        /// Output stream for logging purposes.
        class stream {
        public:
            stream(nlogger& parent, level lvl);

            stream(const stream&) = default;

            template <class T>
            stream& operator<<(const T& x) {
                parent_.log(lvl_, x);
                return *this;
            }

            template <class T>
            stream& operator<<(const optional<T>& x) {
                if (!x)
                    return *this << "-none-";
                return *this << *x;
            }

        private:
            nlogger& parent_;
            level lvl_;
        };

        stream error();
        stream info();
        stream verbose();
        stream massive();

        void disable_colors();

    private:
        nlogger();

        level level_console_;
        level level_file_;
        std::ostream* console_;
        std::ofstream file_;
        std::ostringstream dummy_;
    };
}