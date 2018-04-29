
#include <vector>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>

using namespace std;

struct c_tf_detect_result {
    int nclass;
    float score;
    cv::Rect r;
};

struct c_tf_classify_result {
    int nclass;
    float score;
};

namespace tf_wrapper {

class tf_wrapper{

    public:
        bool init(vector<string>& flags);
        //bool init(string model_name, string model_base_path, string model_config_file);
        static bool preprocess();
        bool detect_predict(const cv::Mat & img, vector<c_tf_detect_result> & output);
        bool classify_predict(const cv::Mat & img, vector<c_tf_classify_result> & output);
        bool postprocess();

    private:

};

}



































