//
//  VolumeDomain.h
//


#ifndef GVT_VOLUME_DOMAIN_H
#define GVT_VOLUME_DOMAIN_H

#include "Domain.h"

#include <GVT/Data/primitives.h>
#include "gvtCell.h"

#include <string>
#include <vector>
using namespace std;

namespace GVT {
    namespace Domain {

        class VolumeDomain : public Domain {
        public:

            VolumeDomain()
            : id(-1), filename(""), data(NULL) {
            }

            VolumeDomain(int id_, string filename_, vector<int>& dim_, vector<float>& min_, vector<float>& max_)
            : id(id_), filename(filename_), dim(dim_), min(min_), max(max_), data(NULL) {
                for (int i = 0; i < 3; ++i) {
                    cell_dim.push_back((max_[i] - min_[i]) / (float) dim[i]);
                }
            }

            VolumeDomain(int id_, string filename_, int* dim_, float* min_, float* max_)
            : id(id_), filename(filename_), data(NULL) {
                for (int i = 0; i < 3; ++i) {
                    dim.push_back(dim_[i]);
                    min.push_back(min_[i]);
                    max.push_back(max_[i]);

                    cell_dim.push_back((max_[i] - min_[i]) / (float) dim[i]);
                }
            }

            VolumeDomain(const VolumeDomain& d) {
                id = d.id;
                filename = d.filename;
                data = d.data; // XXX shallow copy.  Need ref count?

                dim = d.dim;
                min = d.min;
                max = d.max;
            }

            virtual ~VolumeDomain() {
                // data must be explicitly freed with FreeData()
            }

            bool MakeCell(int, gvtCell&);
            virtual bool Intersect(GVT::Data::ray&, vector<int>&);
            bool LoadData();

            void FreeData() {
                delete[] data;
                data = NULL;
            }

            int Size() {
                return dim[0] * dim[1] * dim[2];
            }

            int SizeInBytes() {
                return dim[0] * dim[1] * dim[2] * sizeof (float);
            }

            friend ostream& operator<<(ostream&, VolumeDomain const&);



        private:
            string filename;
            int id;
            vector<int> dim;
            vector<float> min;
            vector<float> max;
            vector<float> cell_dim;
            float * data;
        };

    };
};
#endif // GVT_VOLUME_DOMAIN_H
