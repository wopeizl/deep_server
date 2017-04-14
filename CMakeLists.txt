
#This part is used for compile faster_rcnn_demo.cpp
cmake_minimum_required (VERSION 2.8)
SET(CMAKE_BUILD_TYPE "Debug")  
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")  
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")  

project (deep_server)

if(${CMAKE_SYSTEM_NAME} MATCHES "Window")
  set(WSLIB -lws2_32)
else ()
  set(WSLIB)
endif()

################################################################################
#                                compiler setup                                #
################################################################################

# check for g++ >= 4.8 or clang++ > = 3.2
if(NOT WIN32 AND NOT CAF_NO_COMPILER_CHECK)
  try_run(ProgramResult
          CompilationSucceeded
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_compiler_version.cpp"
          RUN_OUTPUT_VARIABLE CompilerVersion)
  if(NOT CompilationSucceeded OR NOT ProgramResult EQUAL 0)
    message(FATAL_ERROR "Cannot determine compiler version")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    if(CompilerVersion VERSION_GREATER 4.7)
      message(STATUS "Found g++ version ${CompilerVersion}")
    else()
      message(FATAL_ERROR "g++ >= 4.8 required (found: ${CompilerVersion})")
    endif()
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if(CompilerVersion VERSION_GREATER 3.1)
      message(STATUS "Found clang++ version ${CompilerVersion}")
    else()
      message(FATAL_ERROR "clang++ >= 3.2 required (found: ${CompilerVersion})")
    endif()
  else()
    message(FATAL_ERROR "Your C++ compiler does not support C++11 "
                        "or is not supported")
  endif()
endif()
# set optional build flags
set(EXTRA_FLAGS "")
# increase max. template depth on GCC and Clang
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"
   OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  set(EXTRA_FLAGS "-ftemplate-depth=512 -ftemplate-backtrace-limit=0")
endif()
# add "-Werror" flag if --pedantic-build is used
if(CAF_CXX_WARNINGS_AS_ERRORS)
  set(EXTRA_FLAGS "${EXTRA_FLAGS} -Werror")
endif()
# set compiler flags for GCOV if requested
if(CAF_ENABLE_GCOV)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(NO_INLINE "-fno-inline")
  else()
    set(NO_INLINE "-fno-inline -fno-inline-small-functions -fno-default-inline")
  endif()
  set(EXTRA_FLAGS "${EXTRA_FLAGS} -fprofile-arcs -ftest-coverage ${NO_INLINE}")
endif()
# set -fno-exception if requested
if(CAF_FORCE_NO_EXCEPTIONS)
  set(EXTRA_FLAGS "${EXTRA_FLAGS} -fno-exceptions")
endif()
# enable a ton of warnings if --more-clang-warnings is used
if(CAF_MORE_WARNINGS)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(WFLAGS "-Weverything -Wno-c++98-compat -Wno-padded "
               "-Wno-documentation-unknown-command -Wno-exit-time-destructors "
               "-Wno-global-constructors -Wno-missing-prototypes "
               "-Wno-c++98-compat-pedantic -Wno-unused-member-function "
               "-Wno-unused-const-variable -Wno-switch-enum "
               "-Wno-missing-noreturn -Wno-covered-switch-default")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    set(WFLAGS "-Waddress -Wall -Warray-bounds "
               "-Wattributes -Wbuiltin-macro-redefined -Wcast-align "
               "-Wcast-qual -Wchar-subscripts -Wclobbered -Wcomment "
               "-Wconversion -Wconversion-null -Wcoverage-mismatch "
               "-Wcpp -Wdelete-non-virtual-dtor -Wdeprecated "
               "-Wdeprecated-declarations -Wdiv-by-zero -Wdouble-promotion "
               "-Wempty-body -Wendif-labels -Wenum-compare -Wextra "
               "-Wfloat-equal -Wformat -Wfree-nonheap-object "
               "-Wignored-qualifiers -Winit-self "
               "-Winline -Wint-to-pointer-cast -Winvalid-memory-model "
               "-Winvalid-offsetof -Wlogical-op -Wmain -Wmaybe-uninitialized "
               "-Wmissing-braces -Wmissing-field-initializers -Wmultichar "
               "-Wnarrowing -Wnoexcept -Wnon-template-friend "
               "-Wnon-virtual-dtor -Wnonnull -Woverflow "
               "-Woverlength-strings -Wparentheses "
               "-Wpmf-conversions -Wpointer-arith -Wreorder "
               "-Wreturn-type -Wsequence-point -Wshadow "
               "-Wsign-compare -Wswitch -Wtype-limits -Wundef "
               "-Wuninitialized -Wunused -Wvla -Wwrite-strings")
  endif()
  # convert CMake list to a single string, erasing the ";" separators
  string(REPLACE ";" "" WFLAGS_STR ${WFLAGS})
  set(EXTRA_FLAGS "${EXTRA_FLAGS} ${WFLAGS_STR}")
endif()
# add -stdlib=libc++ when using Clang if possible
if(NOT CAF_NO_AUTO_LIBCPP AND "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CXXFLAGS_BACKUP "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++")
  try_run(ProgramResult
          CompilationSucceeded
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_compiler_version.cpp"
          RUN_OUTPUT_VARIABLE CompilerVersion)
  if(NOT CompilationSucceeded OR NOT ProgramResult EQUAL 0)
    message(STATUS "Use clang with GCC' libstdc++")
  else()
    message(STATUS "Automatically added '-stdlib=libc++' flag "
                   "(CAF_NO_AUTO_LIBCPP not defined)")
    set(EXTRA_FLAGS "${EXTRA_FLAGS} -stdlib=libc++")
  endif()
  # restore CXX flags
  set(CMAKE_CXX_FLAGS "${CXXFLAGS_BACKUP}")
endif()
# enable address sanitizer if requested by the user
if(CAF_ENABLE_ADDRESS_SANITIZER)
  # check whether address sanitizer is available
  set(CXXFLAGS_BACKUP "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
  try_run(ProgramResult
          CompilationSucceeded
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_compiler_version.cpp")
  if(NOT CompilationSucceeded)
    message(WARNING "Address Sanitizer is not available on selected compiler")
  else()
    message(STATUS "Enable Address Sanitizer")
    set(EXTRA_FLAGS "${EXTRA_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  endif()
  # restore CXX flags
  set(CMAKE_CXX_FLAGS "${CXXFLAGS_BACKUP}")
endif(CAF_ENABLE_ADDRESS_SANITIZER)
# -pthread is ignored on MacOSX but required on other platforms
if(NOT APPLE AND NOT WIN32)
  set(EXTRA_FLAGS "${EXTRA_FLAGS} -pthread")
endif()
# -fPIC generates warnings on MinGW and Cygwin plus extra setup steps needed on MinGW
if(MINGW)
  add_definitions(-D_WIN32_WINNT=0x0600)
  add_definitions(-DWIN32)
  include(GenerateExportHeader)
  set(LD_FLAGS "ws2_32 -liphlpapi")
  # build static to avoid runtime dependencies to GCC libraries
  set(EXTRA_FLAGS "${EXTRA_FLAGS} -static")
elseif(CYGWIN)
  set(EXTRA_FLAGS "${EXTRA_FLAGS} -U__STRICT_ANSI__")
else()
  set(EXTRA_FLAGS "${EXTRA_FLAGS} -fPIC")
endif()
if (WIN32)
  set(LD_FLAGS ${LD_FLAGS} ws2_32 iphlpapi)
endif()
# iOS support
if(CAF_OSX_SYSROOT)
  set(CMAKE_OSX_SYSROOT "${CAF_OSX_SYSROOT}")
endif()
if(CAF_IOS_DEPLOYMENT_TARGET)
  if("${CAF_OSX_SYSROOT}" STREQUAL "iphonesimulator")
    set(EXTRA_FLAGS "${EXTRA_FLAGS} -mios-simulator-version-min=${CAF_IOS_DEPLOYMENT_TARGET}")
  else()
    set(EXTRA_FLAGS "${EXTRA_FLAGS} -miphoneos-version-min=${CAF_IOS_DEPLOYMENT_TARGET}")
  endif()
endif()
# check if the user provided CXXFLAGS, set defaults otherwise
if(NOT CMAKE_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS                   "-std=c++11 -Wextra -Wall -pedantic ${EXTRA_FLAGS}")
endif()
if(NOT CMAKE_CXX_FLAGS_DEBUG)
  set(CMAKE_CXX_FLAGS_DEBUG             "-O0 -g")
endif()
if(NOT CMAKE_CXX_FLAGS_MINSIZEREL)
  set(CMAKE_CXX_FLAGS_MINSIZEREL        "-Os")
endif()
if(NOT CMAKE_CXX_FLAGS_RELEASE)
  set(CMAKE_CXX_FLAGS_RELEASE           "-O3 -DNDEBUG")
endif()
if(NOT CMAKE_CXX_FLAGS_RELWITHDEBINFO)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO    "-O2 -g")
endif()
# set build default build type to RelWithDebInfo if not set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
# needed by subprojects
set(LD_FLAGS ${LD_FLAGS} ${CMAKE_LD_LIBS})


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

#include(cmake/Utils.cmake)
#include(cmake/Cuda.cmake)
# packages                                                                                                                                                                                                 
find_package(CUDA)

set(DEEP_ROOT /data1/zhangyuanpeng/workspace/online_serving/SDK/deep_server/)

set(Deep_server_LINKER_LIBS "")
set(Deep_server_INCLUDE_DIRS "")
set(Deep_server_DEFINITIONS "")
set(Deep_server_COMPILE_OPTIONS "")

# ---[ Boost
#find_package(Boost 1.46 REQUIRED COMPONENTS system thread filesystem)
#list(APPEND Deep_server_INCLUDE_DIRS PUBLIC ${Boost_INCLUDE_DIRS})
#list(APPEND Deep_server_DEFINITIONS PUBLIC -DBOOST_ALL_NO_LIB)
#list(APPEND Deep_server_LINKER_LIBS PUBLIC ${Boost_LIBRARIES})

# nvcrc flags                                                                                                                                                                                               
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
if(NOT MINGW)
  set(CAF_ROOT ${DEEP_ROOT}/3rd/actor-framework)
  set(JSONCPP_ROOT ${DEEP_ROOT}/Depends/jsoncpp)
  set(CAFFE_ROOT ${DEEP_ROOT}/Depends/caffe)
else()
  set(CAF_ROOT d:/work/deep_server/3rd/actor-framework)
endif()

	set(DEEP_SRCS src/utils.cpp
			src/actorlogger.cpp
			src/base64.cpp
			src/caffe_process.cpp
			src/cv_process.cpp
			src/deep_server.cpp
			src/http_broker.cpp
			src/http_parser.cpp
			src/parserhelper.cpp
			src/tcp_broker.cpp
	   )

add_executable(deep_server ${DEEP_SRCS})

if(${CMAKE_SYSTEM_NAME} MATCHES "Window")
  set(WSLIB -lws2_32)
else ()
  set(WSLIB)
endif()

target_link_libraries(deep_server
        ${LD_FLAGS}
        ${CAF_LIBRARIES}
        ${PTHREAD_LIBRARIES}
        ${WSLIB})

# boost support
  set(Boost_USE_MULTITHREADED ON)
  find_package(Boost COMPONENTS system filesystem)
  if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set(LD_DIRS ${LD_DIRS} ${Boost_LIBRARIES})
    set(LD_FLAGS ${LD_FLAGS} ${Boost_SYSTEM_LIBRARY}  ${Boost_FILESYSTEM_LIBRARY}) 
  endif()

#opencv support
  find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
  if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
  endif()
  list(APPEND Deep_server_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
  list(APPEND Deep_server_LINKER_LIBS ${OpenCV_LIBS})
  message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  list(APPEND Deep_server_DEFINITIONS -DUSE_OPENCV)

if(NOT MINGW)
  include_directories ( "${PROJECT_SOURCE_DIR}include"
          "${PROJECT_SOURCE_DIR}/src"
          /usr/local/include
          /usr/include
          /usr/include/openblas
          /usr/include/python2.7
          /usr/local/cuda/include
          "${JSONCPP_ROOT}/include"
          "${CAFFE_ROOT}/include"
          "${CAF_ROOT}/libcaf_io"
          "${CAF_ROOT}/libcaf_core"
#          "${Deep_server_INCLUDE_DIRS}"
          )
  
target_link_libraries(deep_server
          /lib64/libgflags.so
          "${JSONCPP_ROOT}/lib/libjsoncpp.a"
          /lib64/libopencv_highgui.so
          /lib64/libopencv_core.so
          /lib64/libopencv_imgproc.so
          /lib64/libglog.so
          /lib64/libboost_system.so
          /lib64/libboost_python.so
          /lib64/libpython2.7.so
          "${CAFFE_ROOT}/lib/libcaffe.so"
          "${CAF_ROOT}/build/lib/libcaf_core.so"
          "${CAF_ROOT}/build/lib/libcaf_io.so"
 #         "${Deep_server_LINKER_LIBS}"
          )
else()

  include_directories ( "${PROJECT_SOURCE_DIR}/include"
          "${PROJECT_SOURCE_DIR}/src"
          "${CAF_ROOT}/libcaf_io"
          "${CAF_ROOT}/libcaf_core"
          )

  target_link_libraries(deep_server
          "${PROJECT_SOURCE_DIR}/build/lib/libcaf_core_static.a"
          "${PROJECT_SOURCE_DIR}/build/lib/libcaf_io_static.a"
          )
endif()



