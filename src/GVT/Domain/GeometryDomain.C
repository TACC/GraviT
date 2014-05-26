//
// GeometryDomain.C
//


#include "GeometryDomain.h"

namespace GVT {
    namespace Domain {

        void GeometryDomain::FreeData() {
            if (!domain_loaded) return;
            for (int i = lights.size()-1; i >= 0; i--) {
                delete lights[i];
                lights.pop_back();
            }
            if (mesh->mat) {
                delete mesh->mat;
                mesh->mat = NULL;
            }
            if(mesh) {
                delete mesh;
                mesh = NULL;
            }
            domain_loaded = false;
        }

        bool GeometryDomain::Intersect(GVT::Data::ray& ray, vector<int>& intersections) {
            bool hit = boundingBox.intersect(ray);
            if (!hit) {
                intersections.clear();
                return false;
            } else {
                return true;
            }
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