
#include "tf_wrapper.h"
using namespace std;

namespace tf_wrapper{ 

    bool tf_wrapper::init(vector<string>& flags) {
        return false;
    }

    bool tf_wrapper::preprocess() {
        return false;
    }

    bool tf_wrapper::predict(vector<unsigned char>& idata, void *p) {
        return false;
    }

    bool tf_wrapper::postprocess() {
        return false;
    }
}

