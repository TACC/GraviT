//
// Dataset.h
//

#ifndef GVT_RENDER_DATA_DATASET_H
#define GVT_RENDER_DATA_DATASET_H


#include <gvt/core/Debug.h>
#include <gvt/render/data/Domains.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/Camera.h>
#include <gvt/render/data/scene/gvtCamera.h>
#include <gvt/render/data/scene/Light.h>
#include <gvt/render/data/accel/AbstractAccel.h>

#include <algorithm>
#include <cfloat>
#include <map>
#include <string>
#include <vector>
#include <map>

namespace gvt {
    namespace render {
        class Attributes;
        namespace data {
            class Dataset {
            public:

                Dataset();
                ~Dataset();

                virtual bool init();
                virtual int size();
                virtual bool intersect(gvt::render::actor::Ray& r, gvt::render::actor::isecDomList& inter);
                virtual gvt::render::data::domain::AbstractDomain* getDomain(int id);
                virtual gvt::render::data::scene::Light* getLight(int id);
                virtual int addDomain(gvt::render::data::domain::AbstractDomain* dom);
                virtual int addLight(gvt::render::data::scene::Light* ls);
                void makeAccel(gvt::render::Attributes& rta);
				void makeAccel();

                friend std::ostream& operator<<(std::ostream&, Dataset const&);

//        protected:
            public:
                gvt::render::data::primitives::Box3D                        dataSetBB;
                gvt::render::data::scene::Camera                            camera;
                gvt::render::data::scene::gvtPerspectiveCamera              GVTCamera;
                std::vector<gvt::render::data::domain::AbstractDomain*>     domainSet;
                std::vector<gvt::render::data::scene::Light*>               lightSet;
                gvt::render::data::accel::AbstractAccel*                    acceleration;
                std::map<std::string, gvt::render::data::primitives::Mesh*> objMeshes;
            };
        }
    }
}

#endif // GVT_RENDER_DATA_DATASET_H
