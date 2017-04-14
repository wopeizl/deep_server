#pragma once

#include <iterator>  
#include <vector>  
#include <string>  
#include <algorithm>  
#include <random>  
#include <ctime>

#if defined WIN32 || defined WINCE
#include "opencv2/imgproc.hpp"
#else
#include "opencv2/imgproc/imgproc.hpp"
#endif
#include "opencv2/imgproc/imgproc_c.h"

#include "caffe_config.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <ctype.h>
#include <assert.h>

#if defined WIN32 || defined WINCE
#include <windows.h>
#if !defined _WIN32_WINNT
#ifdef HAVE_MSMF
#define _WIN32_WINNT 0x0600 // Windows Vista
#else
#define _WIN32_WINNT 0x0501 // Windows XP
#endif
#endif

#undef small
#undef min
#undef max
#undef abs
#endif

using namespace std;
using namespace cv;

class cvprocess {
public:

    static bool flip(std::vector<unsigned char>& idata, std::vector<unsigned char>& odata);
    static bool medianBlur(std::vector<unsigned char>& idata, std::vector<unsigned char>& odata);
    static bool applyColorMap(std::vector<unsigned char>& idata, std::vector<unsigned char>& odata);
    static bool cvtColor(std::vector<unsigned char>& idata, std::vector<unsigned char>& odata);
    static bool readImage(std::vector<unsigned char>& idata, cv::Mat& dst);
    static bool writeImage(cv::Mat& src, std::vector<unsigned char>& odata);
    static bool process_caffe_result(std::vector<caffe::Frcnn::BBox<float>> results, cv::Mat& dst);
};

