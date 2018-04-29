
#include "cv_process.hpp"
#include <fstream>

bool cvprocess::flip(std::vector<unsigned char>& idata, std::vector<unsigned char>& odata) {
    Mat src;
    cv::Mat dst;

    if (!readImage(idata, src))
        return false;

    cv::flip(src, dst, 0);

    return writeImage(dst, odata);
}

bool cvprocess::medianBlur(std::vector<unsigned char>& idata, std::vector<unsigned char>& odata) {
    Mat src;
    cv::Mat dst;

    if (!readImage(idata, src))
        return false;

    cv::medianBlur(src, dst, 3);

    return writeImage(dst, odata);
}

bool cvprocess::applyColorMap(std::vector<unsigned char>& idata, std::vector<unsigned char>& odata) {
    Mat src;
    cv::Mat dst;

    if (!readImage(idata, src))
        return false;

    //cv::applyColorMap(src, dst, 0);

    return writeImage(dst, odata);
}

bool cvprocess::cvtColor(std::vector<unsigned char>& idata, std::vector<unsigned char>& odata) {
    Mat src;
    cv::Mat dst;

    if (!readImage(idata, src))
        return false;

    cv::cvtColor(src, dst, CV_BGR2GRAY);

    return writeImage(dst, odata);
}


bool cvprocess::readImage(std::vector<unsigned char>& idata, cv::Mat& dst) {
    if (idata.size() == 0)
        return false;

    dst = cv::imdecode(idata, CV_LOAD_IMAGE_UNCHANGED);

    return dst.data != nullptr;
}

bool cvprocess::writeImage(cv::Mat& src, std::vector<unsigned char>& odata) {
    if (src.data == nullptr)
        return false;

    vector<int> param;
    param.push_back(IMWRITE_PNG_COMPRESSION);
    param.push_back(3); //default(3) 0-9.
    return cv::imencode(".png", src, odata, param);
}

bool cvprocess::process_caffe_result(std::vector<caffe::Frcnn::BBox<float>> results, cv::Mat& cv_image) {
    for (int ir = 0; ir < results.size(); ir++) {
        if (results[ir].confidence > 0.9
            /*&& results[ir].id == 7*/) {
            cv::Rect rect(results[ir][0], results[ir][1], results[ir][2], results[ir][3]);
           // cv::rectangle(cv_image, rect, cv::Scalar(255, 0, 0), 2);

           cv::rectangle(cv_image, cv::Point(results[ir][0],results[ir][1])
                       , cv::Point(results[ir][2],results[ir][3]), cv::Scalar(255, 0, 0), 2);
            char text[64];
            sprintf(text, "%lf", results[ir].confidence);
            cv::putText(cv_image
                , text
                , cv::Point(rect.x, rect.y)
                , CV_FONT_HERSHEY_COMPLEX
                , 0.8
                , cv::Scalar(0, 0, 255));
        }
    }

    return true;
}

bool cvprocess::process_tf_detect_result(std::vector<deep_server::Tf_detect_result> results, cv::Mat& cv_image) {
    for (int ir = 0; ir < results.size(); ir++) {
        //if (results[ir].confidence > 0.9
            ///*&& results[ir].id == 7*/) 
        {
            cv::Rect rect(results[ir].x(), results[ir].y(), results[ir].w(), results[ir].h());
           // cv::rectangle(cv_image, rect, cv::Scalar(255, 0, 0), 2);

           cv::rectangle(cv_image, cv::Point(results[ir].x(), results[ir].y())
                       , cv::Point(results[ir].w(), results[ir].h()), cv::Scalar(255, 0, 0), 2);
            char text[64];
            sprintf(text, "%lf", results[ir].score());
            cv::putText(cv_image
                , text
                , cv::Point(rect.x, rect.y)
                , CV_FONT_HERSHEY_COMPLEX
                , 0.8
                , cv::Scalar(0, 0, 255));
        }
    }

    return true;
}
