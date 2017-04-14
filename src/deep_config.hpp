
#include "caf/config.hpp"

#ifdef WIN32
# define _WIN32_WINNT 0x0600
# include <Winsock2.h>
#else
# include <arpa/inet.h> // htonl
#endif


#include <vector>
#include <string>
#include <limits>
#include <memory>
#include <cstdint>
#include <cassert>
#include <iostream>
#include <fstream>
#include <csignal>
#include <ctime>
#include <regex>
#include <cctype>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <condition_variable>

#include "caf/all.hpp"
#include "caf/io/all.hpp"

#include "caf/config.hpp"

#if defined(CAF_LINUX) || defined(CAF_MACOS)
#include <unistd.h>
#include <cxxabi.h>
#include <sys/types.h>
#endif

#include "caf/all.hpp"
#include "caf/io/all.hpp"
#include "actorlogger.hpp"

#include "caf/string_algorithms.hpp"

#include "caf/term.hpp"
#include "caf/locks.hpp"
#include "caf/timestamp.hpp"
#include "caf/actor_proxy.hpp"
#include "caf/actor_system.hpp"

#include "caf/detail/get_process_id.hpp"
#include "caf/detail/single_reader_queue.hpp"

using std::cout;
using std::cerr;
using std::endl;

using namespace caf;
using namespace caf::io;
using namespace std;

#ifndef UNUSED
#define UNUSED(expr) do { (void)(expr); } while (0)
#endif
#ifdef __clang__
# pragma clang diagnostic ignored "-Wshorten-64-to-32"
# pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
# pragma clang diagnostic ignored "-Wunused-const-variable"
# pragma clang diagnostic ignored "-Wunused-parameter"

#endif // __clang__

#define CAF_LOG_LEVEL 

#if defined(CAF_LINUX) || defined(CAF_MACOS)
#else
#pragma warning( disable : 4819)
#pragma warning( disable : 4267)
#endif

#define NO_STRICT 
