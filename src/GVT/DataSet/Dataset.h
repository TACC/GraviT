//
// Dataset.h
//

#ifndef GVT_DATASET_H
#define GVT_DATASET_H


#include <GVT/Data/primitives.h>
#include <GVT/Domain/domains.h>

#include <cfloat>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

#include <GVT/common/debug.h>

using namespace std;

namespace GVT {
    namespace Dataset {

        class GVTDataset {

            
        public:

            GVTDataset() {
            }

            virtual bool init();
            virtual int size();
            virtual bool intersect(GVT::Data::ray&  r, GVT::Data::isecDomList& inter);
            virtual GVT::Domain::Domain* getDomain(int id);
            virtual GVT::Data::LightSource* getLightSource(int id);
            virtual int addDomain(GVT::Domain::Domain* dom);
            virtual int addLightSource(GVT::Data::LightSource* ls);
            
            
            friend ostream& operator<<(ostream&, GVTDataset const&);

        protected:
            GVT::Data::box3D dataSetBB;
            std::vector<GVT::Domain::Domain*> domainSet;
            std::vector<GVT::Data::LightSource*> lightSet; 
        };
    };
};

#endif // GVT_DATASET_H
