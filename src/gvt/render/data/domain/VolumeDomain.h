//
//  VolumeDomain.h
//


#ifndef GVT_RENDER_DATA_DOMAIN_VOLUME_DOMAIN_H
#define GVT_RENDER_DATA_DOMAIN_VOLUME_DOMAIN_H

#include <gvt/render/data/domain/AbstractDomain.h>
#include <gvt/render/data/domain/Cell.h>

#include <gvt/render/data/Primitives.h>

#include <string>
#include <vector>

namespace gvt {
    namespace render {
        namespace data {
            namespace domain {
                /// atomic volumetric data unit for GraviT internal use
                /** Domain for volume data. 
                \sa AbstractDomain, GeometryDomain
                */
                class VolumeDomain : public AbstractDomain 
                {
                public:

                    VolumeDomain() 
                    : id(-1), filename(""), data(NULL) 
                    {}

                    VolumeDomain(int id_, std::string filename_, std::vector<int>& dim_, std::vector<float>& min_, std::vector<float>& max_)
                    : id(id_), filename(filename_), dim(dim_), min(min_), max(max_), data(NULL) 
                    {
                        for (int i = 0; i < 3; ++i) 
                        {
                            cell_dim.push_back((max_[i] - min_[i]) / (float) dim[i]);
                        }
                    }

                    VolumeDomain(int id_, std::string filename_, int* dim_, float* min_, float* max_)
                    : id(id_), filename(filename_), data(NULL) 
                    {
                        for (int i = 0; i < 3; ++i) 
                        {
                            dim.push_back(dim_[i]);
                            min.push_back(min_[i]);
                            max.push_back(max_[i]);

                            cell_dim.push_back((max_[i] - min_[i]) / (float) dim[i]);
                        }
                    }

                    VolumeDomain(const VolumeDomain& d) 
                    {
                        id = d.id;
                        filename = d.filename;
                        data = d.data; // XXX shallow copy.  Need ref count?

                        dim = d.dim;
                        min = d.min;
                        max = d.max;
                    }

                    virtual ~VolumeDomain() 
                    {
                        // data must be explicitly freed with FreeData()
                    }

                    bool MakeCell(int, Cell&);
                    virtual bool intersect(gvt::render::actor::Ray&, std::vector<int>&);
                    bool load();

                    void free() 
                    {
                        delete[] data;
                        data = NULL;
                    }

                    int size() 
                    {
                        return dim[0] * dim[1] * dim[2];
                    }

                    int sizeInBytes() 
                    {
                        return dim[0] * dim[1] * dim[2] * sizeof (float);
                    }

                    friend std::ostream& operator<<(std::ostream&, VolumeDomain const&);

                private:
                    std::string filename;
                    int id;
                    std::vector<int> dim;
                    std::vector<float> min;
                    std::vector<float> max;
                    std::vector<float> cell_dim;
                    float * data;
                };
            }
        }
    }
}

#endif // GVT_RENDER_DATA_DOMAIN_VOLUME_DOMAIN_H
