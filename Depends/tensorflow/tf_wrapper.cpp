
#include "tf_wrapper.h"
using namespace std;

namespace tf_wrapper{ 

    bool tf_wrapper::init(vector<string>& flags) {
        return false;
    }

    bool tf_wrapper::init(string model_name, string model_base_path, string model_config_file){
        return false;
    }

    bool tf_wrapper::preprocess() {
        return false;
    }

    bool tf_wrapper::predict(vector<unsigned char>& idata, vector<p_tf_detect_result> & output) {
        return false;
    }

    bool tf_wrapper::postprocess() {
        return false;
    }
}

