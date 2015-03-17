//
// Dataset.h
//

#ifndef GVT_RENDER_DATA_DATASET_H
#define GVT_RENDER_DATA_DATASET_H


#include <gvt/core/Debug.h>
#include <gvt/render/data/Domains.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/Camera.h>
#include <gvt/render/data/scene/Light.h>

#include <algorithm>
#include <cfloat>
#include <map>
#include <string>
#include <vector>

namespace gvt {
    namespace render {
        namespace data {
            class Dataset {
            public:

                Dataset() {}

                virtual bool init();
                virtual int size();
                virtual bool intersect(gvt::render::actor::Ray& r, gvt::render::actor::isecDomList& inter);
                virtual gvt::render::data::domain::AbstractDomain* getDomain(int id);
                virtual gvt::render::data::scene::Light* getLight(int id);
                virtual int addDomain(gvt::render::data::domain::AbstractDomain* dom);
                virtual int addLight(gvt::render::data::scene::Light* ls);


                friend std::ostream& operator<<(std::ostream&, Dataset const&);

//        protected:
            public:
                gvt::render::data::primitives::Box3D                        dataSetBB;
                gvt::render::data::scene::Camera                            camera;
                std::vector<gvt::render::data::domain::AbstractDomain*>     domainSet;
                std::vector<gvt::render::data::scene::Light*>               lightSet; 
            };
        }
    }
}

#endif // GVT_RENDER_DATA_DATASET_H
