
#include <vector>
#include <string>
using namespace std;

namespace tf_wrapper {

class tf_wrapper{

    public:
        bool init(vector<string>& flags);
        bool preprocess();
        bool predict(vector<unsigned char>& idata, void* p);
        bool postprocess();

    private:

};

}



































