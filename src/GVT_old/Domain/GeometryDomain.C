//
// GeometryDomain.C
//


#include "GeometryDomain.h"
#include <boost/timer/timer.hpp>

namespace GVT {
    namespace Domain {

        void GeometryDomain::free() {
            if (!isLoaded) return;
            for (int i = lights.size() - 1; i >= 0; i--) {
                delete lights[i];
                lights.pop_back();
            }
            if (mesh->mat) {
                delete mesh->mat;
                mesh->mat = NULL;
            }
            if (mesh) {
                delete mesh;
                mesh = NULL;
            }
            isLoaded = false;
        }

//        bool GeometryDomain::intersect(GVT::Data::ray& ray, vector<int>& intersections) {
//            if (!boundingBox.intersect(ray)) {
//                intersections.clear();
//                return false;
//            } else {
//                return true;
//            }
//        }

        bool GeometryDomain::load() {
            if (isLoaded) return true;
            if (filename == "") return false;

            {
              printf("loading file\n");
              boost::timer::auto_cpu_timer t;
            mesh = readply(filename);
            // mesh = new GVT::Data::Mesh;
            }

            lights.push_back(new GVT::Data::PointLightSource(GVT::Math::Point4f(5.0, 5.0, 5.0, 1.f), GVT::Data::Color(1.f, 1.f, 1.f, 1.f)));
            mesh->mat = new GVT::Data::Lambert(GVT::Data::Color(1.f, .0f, .0f, 1.f));
            boundingBox = mesh->boundingBox;
            isLoaded = true;
            return isLoaded;
        }

        ostream&
        operator<<(ostream& os, GeometryDomain const& d) {
            os << "geometry domain @ addr " << (void*) &d << endl;
            os << "    XXX not yet implemented XXX" << endl;
            os << flush;

            return os;
        }
    }
}
