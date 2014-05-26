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

#include <GVT/common/debug.h>

using namespace std;

namespace GVT {
    namespace Dataset {

        class abstract_dataset {
        public:

            abstract_dataset() {
            }

            abstract_dataset(string& filename) : conf_filename(filename) {
            }

            virtual bool Init() {
                 GVT_DEBUG(DBG_ALWAYS,"Abstract load");
                 return false;
            }; 
            virtual int Size() = 0;
            virtual bool Intersect(GVT::Data::ray&, vector<int>&) = 0;
            virtual GVT::Domain::Domain* GetDomain(int) = 0;
            friend ostream& operator<<(ostream&, abstract_dataset const&);

        protected:
            string conf_filename;
        };
    };
};

#endif // GVT_DATASET_H
