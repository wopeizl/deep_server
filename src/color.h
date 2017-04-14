//
// Created by Administrator on 2017/4/5.
//

#ifndef DEEP_SERVER_COLOR_H
#define DEEP_SERVER_COLOR_H

namespace color {

// UNIX terminal color codes
    constexpr char reset[]        = "\033[0m";
    constexpr char reset_endl[]   = "\033[0m\n";
    constexpr char black[]        = "\033[30m";
    constexpr char red[]          = "\033[31m";
    constexpr char green[]        = "\033[32m";
    constexpr char yellow[]       = "\033[33m";
    constexpr char blue[]         = "\033[34m";
    constexpr char magenta[]      = "\033[35m";
    constexpr char cyan[]         = "\033[36m";
    constexpr char white[]        = "\033[37m";
    constexpr char bold_black[]   = "\033[1m\033[30m";
    constexpr char bold_red[]     = "\033[1m\033[31m";
    constexpr char bold_green[]   = "\033[1m\033[32m";
    constexpr char bold_yellow[]  = "\033[1m\033[33m";
    constexpr char bold_blue[]    = "\033[1m\033[34m";
    constexpr char bold_magenta[] = "\033[1m\033[35m";
    constexpr char bold_cyan[]    = "\033[1m\033[36m";
    constexpr char bold_white[]   = "\033[1m\033[37m";

} // namespace color

#endif //DEEP_SERVER_COLOR_H
