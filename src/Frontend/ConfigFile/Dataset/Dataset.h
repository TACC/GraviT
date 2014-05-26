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
        class Dataset : public abstract_dataset {
        public:

            Dataset() {
            }

            Dataset(string& filename) : abstract_dataset(filename) {
                GVT_DEBUG(DBG_ALWAYS, "Filename : " + filename);
                conf_filename = filename;
            }

            virtual bool Init() {
                GVT_DEBUG(DBG_ALWAYS,"Generic load");
                return false;
            }

            int Size() {
                return files.size();
            }

            virtual bool Intersect(GVT::Data::ray& r, vector<int>& domlist) {
                std::map<int, GVT::Data::box3D > ::iterator itDom = dom_bbox.begin();
                std::set<float> mtmin;
                std::map<float, int> values;

                for (; itDom != dom_bbox.end(); itDom++) {
                    GVT::Data::box3D bbox = itDom->second;
                    int idx = itDom->first;
                    float tmin = FLT_MAX, tmax;
                    if (bbox.intersect(r, tmin, tmax)) {
                        if (tmin > 0)
                            mtmin.insert(tmin);
                        else
                            mtmin.insert(tmax);
                        values[tmin] = idx;
                        GVT_DEBUG(DBG_LOW, "Intersected here " + to_string(idx));
                    }
                }

                if (r.type == GVT::Data::ray::PRIMARY) {
                    for (std::set<float>::iterator i = mtmin.begin(); i != mtmin.end(); i++) {
                        domlist.push_back(values[*i]);
                    }
                } else {
                    for (std::set<float>::reverse_iterator i = mtmin.rbegin(); i != mtmin.rend(); i++) {
                        domlist.push_back(values[*i]);
                    }
                }
                return (!domlist.empty());
            }

            virtual GVT::Domain::Domain* GetDomain(int id) {
                GVT_ASSERT_BACKTRACE("Not implemented!", 1);
                return NULL;
            };

            //    friend ostream& operator<<(ostream&, DomainType const&);

        private:
            vector<string> files;
            vector< vector<int> > sizes;
            vector< vector<int> > offsets;

            map<int, DomainType> dom_cache; // XXX TODO fix when this made subclass
            map<int, GVT::Data::box3D > dom_bbox;
            map<int, GVT::Math::AffineTransformMatrix<float> > dom_model;
            string conf_filename;

            float min[3];
            float max[3];
            int size[3];
        };

        template<> bool Dataset<GVT::Domain::VolumeDomain>::Init();
        template<> GVT::Domain::Domain* Dataset<GVT::Domain::VolumeDomain>::GetDomain(int id);
        
        template<> bool Dataset<GVT::Domain::GeometryDomain>::Init();
        template<> GVT::Domain::Domain* Dataset<GVT::Domain::GeometryDomain>::GetDomain(int id);

        template<> bool Dataset<GVT::Domain::MantaDomain>::Init();
        template<> GVT::Domain::Domain* Dataset<GVT::Domain::MantaDomain>::GetDomain(int id); 
        
    };
};

#endif // GVT_DATASET_H
