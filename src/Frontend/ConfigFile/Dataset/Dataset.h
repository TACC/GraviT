//
// Dataset.h
//

#ifndef GVT_CF_DATASET_H
#define GVT_CF_DATASET_H


#include <GVT/Data/primitives.h>
#include <GVT/Domain/domains.h>
#include <Backend/Manta/Domain/MantaDomain.h>
#include <cfloat>
#include <map>
#include <string>
#include <vector>

#include <GVT/common/debug.h>
#include <GVT/DataSet/Dataset.h>
using namespace std;

namespace GVT {
    namespace Dataset {

        template<typename DomainType>
        class Dataset : public GVTDataset {
        public:

            Dataset() {
            }

            Dataset(string& filename) : GVTDataset(), conf_filename(filename) {
                GVT_DEBUG(DBG_ALWAYS, "Filename : " + filename);
                conf_filename = filename;
            }

            virtual bool init() {
                GVT_DEBUG(DBG_ALWAYS,"Generic load");
                return false;
            }
            
        private:
            vector<string> files;
//            map<int, DomainType> dom_cache; // XXX TODO fix when this made subclass
//            map<int, GVT::Data::box3D > dom_bbox;
//            map<int, GVT::Math::AffineTransformMatrix<float> > dom_model;
            string conf_filename;


        };

        template<> bool Dataset<GVT::Domain::VolumeDomain>::init();
//        template<> GVT::Domain::Domain* Dataset<GVT::Domain::VolumeDomain>::getDomain(int id);
        
        template<> bool Dataset<GVT::Domain::GeometryDomain>::init();
//        template<> GVT::Domain::Domain* Dataset<GVT::Domain::GeometryDomain>::getDomain(int id);

        template<> bool Dataset<GVT::Domain::MantaDomain>::init();
//        template<> GVT::Domain::Domain* Dataset<GVT::Domain::MantaDomain>::getDomain(int id);
        
    };
};

#endif // GVT_DATASET_H
