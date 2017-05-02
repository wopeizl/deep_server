cmake_minimum_required(VERSION 2.8.7)
if(MSVC)
    # CMake 3.4 introduced a WINDOWS_EXPORT_ALL_SYMBOLS target property that makes it possible to
    # build shared libraries without using the usual declspec() decoration.
    # See: https://blog.kitware.com/create-dlls-on-windows-without-declspec-using-new-cmake-export-all-feature/
    # and https://cmake.org/cmake/help/v3.5/prop_tgt/WINDOWS_EXPORT_ALL_SYMBOLS.html
    # for details.
    cmake_minimum_required(VERSION 3.4)
endif()
if(POLICY CMP0046)
    cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

# ---[ deep_server project
project(deep_server C CXX)

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
    src/yolo_process.cpp
    )

add_executable(deep_server ${DEEP_SRCS}) 

# ---[ deep_server version
set(DEEP_TARGET_VERSION "1.0.0-rc5" CACHE STRING "Deep logical version")
set(DEEP_TARGET_SOVERSION "1.0.0-rc5" CACHE STRING "Deep soname version")
add_definitions(-DDEEP_VERSION=${DEEP_TARGET_VERSION} -DGLOG_NO_ABBREVIATED_SEVERITIES)

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(cmake/CompilerCheck.cmake)
include(cmake/Utils.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)

if(MSVC)
    deep_option(CPU_ONLY  "Build deep with CUDA support" ON) 
    deep_option(USE_CUDA  "Build deep without CUDA support" OFF) 
    add_definitions(-DCPU_ONLY -DBOOST_ALL_NO_LIB)
else()
    deep_option(CPU_ONLY  "Build deep without CUDA support" OFF)
    deep_option(USE_CUDA  "Build deep with CUDA support" ON) 
endif()
deep_option(USE_CUDNN "Build deep with cuDNN library support" ON IF NOT CPU_ONLY)
deep_option(USE_OPENCV "Build with OpenCV support" ON)
deep_option(USE_OPENMP "Link with OpenMP (when your BLAS wants OpenMP and you get linker errors)" OFF)
deep_option(USE_NCCL "Build deep with NCCL library support" OFF)

include(cmake/Cuda.cmake)
include("cmake/External/glog.cmake")
    include_directories(${GLOG_INCLUDE_DIRS})
    target_link_libraries(deep_server ${GLOG_LIBRARIES} )

# ---[ Google-gflags
include("cmake/External/gflags.cmake")
    include_directories(${GFLAGS_INCLUDE_DIRS})
    target_link_libraries(deep_server ${GFLAGS_LIBRARIES} )

# ---[ Flags
if(UNIX OR APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

if(USE_libstdcpp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
    message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

# ---[ Warnings
deep_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Includes
#set(Deep_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(deep_server_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})

if(MSVC)
    include("c:/Users/Administrator/.caffe/dependencies/libraries_v140_x64_py27_1.1.0/libraries/caffe-builder-config.cmake")
endif()

if(MSVC)
    include_directories("d:/work/caffe-windows/caffe/include/")
    include_directories("d:/work/caffe-windows/caffe/build/include/")
    include_directories("d:/work/deep_server/3rd/actor-framework/libcaf_io/")
    include_directories("d:/work/deep_server/3rd/actor-framework/libcaf_core/")
    include_directories("d:/work/deep_server/3rd/jsoncpp-master/include/")
    include_directories("d:/work/auto_yolo/src/")
    link_directories("d:/work/deep_server/3rd/jsoncpp-master/build/src/lib_json/${CMAKE_BUILD_TYPE}/")
    link_directories("d:/work/deep_server/3rd/actor-framework/build/lib/${CMAKE_BUILD_TYPE}/")

    target_link_libraries(deep_server
        "d:/work/deep_server/3rd/actor-framework/build64/lib/Debug/caf_core_static.lib"
        "d:/work/deep_server/3rd/actor-framework/build64/lib/Debug/caf_io_static.lib"
        "d:/work/caffe-windows/caffe/build/lib/Debug/caffe-d.lib"
        "d:/work/deep_server/3rd/jsoncpp-master/build/src/lib_json/Debug/jsoncpp.lib"
        ws2_32
        )
else()
    include_directories("${PROJECT_SOURCE_DIR}/3rd/auto_caffe/include/")
    include_directories("${PROJECT_SOURCE_DIR}/3rd/auto_caffe/build/include/")
    include_directories("${PROJECT_SOURCE_DIR}/3rd/actor-framework/libcaf_io/")
    include_directories("${PROJECT_SOURCE_DIR}/3rd/actor-framework/libcaf_core/")
    include_directories("${PROJECT_SOURCE_DIR}/3rd/jsoncpp/include/")
    include_directories("${PROJECT_SOURCE_DIR}/3rd/auto_yolo/src/")
    include_directories (
        /usr/local/include
        /usr/include
        /usr/include/openblas
        /usr/include/python2.7
        /usr/local/cuda/include
        )
    link_directories("${PROJECT_SOURCE_DIR}/3rd/jsoncpp-master/build/src/lib_json/")
    link_directories("${PROJECT_SOURCE_DIR}/3rd/actor-framework/build/lib/")
    set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} 
        "/usr/local/lib"
        "/usr/local/lib64")

    target_link_libraries(deep_server
        "${PROJECT_SOURCE_DIR}/3rd/actor-framework/build/lib/libcaf_core.so"
        "${PROJECT_SOURCE_DIR}/3rd/actor-framework/build/lib/libcaf_io.so"
        "${PROJECT_SOURCE_DIR}/3rd/auto_caffe/build/lib/libcaffe.so"
        "${PROJECT_SOURCE_DIR}/3rd/jsoncpp/build/src/lib_json/libjsoncpp.a"
        "${PROJECT_SOURCE_DIR}/3rd/auto_yolo/lib/libdnl.a"
        #/lib64/libgflags.so
        #/usr/lib64/libglog.so
        )
endif()
 

# boost support
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS system filesystem date_time)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif() 

find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
endif()
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
endif() 

include(cmake/ProtoBuf.cmake)

if(MSVC)
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/ignore:4099")
endif()


# ---[ Configuration summary
deep_print_configuration_summary()

