// gvt2 data tests
//
#include <iostream>
#include "Ray.h"
using namespace std;

int main(int argc,char* argv[]) {
    gvt2::vec3f origin(1.0,2.0,3.0);
    gvt2::vec3f direction(1.0,1.0,1.0);
    gvt2::Ray myray(origin,direction);
    cerr<<"testing Rays " << endl;
    return 1;
}
