
#include <vector>
#include <string>
using namespace std;

    struct p_tf_detect_result {
        int nclass;
        float score;
        float x, y, w, h;
    };

namespace tf_wrapper {

class tf_wrapper{

    public:
        bool init(vector<string>& flags);
        bool init(string model_name, string model_base_path, string model_config_file);
        bool preprocess();
        bool predict(vector<unsigned char>& idata, vector<p_tf_detect_result> & output);
        bool postprocess();

    private:

};

}



































