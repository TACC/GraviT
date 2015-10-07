//
// Domain.h
//

#ifndef GVT_RENDER_DATA_DOMAIN_ABSTRACT_DOMAIN_H
#define GVT_RENDER_DATA_DOMAIN_ABSTRACT_DOMAIN_H

#include <gvt/render/data/Primitives.h>
#include <gvt/core/Math.h>
#include <vector>

#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/mutex.hpp>

namespace gvt {
    namespace render {
        namespace data {
            namespace domain {
                /// base class for data domains
                /** base class for GraviT data domains. A domain is the atomic data representation within GraviT. 
                A domain is passed to a rendering engine in its entirety, where it might be subdivided for the engine's use.
                Domains are not subdivided within GraviT.
                */
                class AbstractDomain {
                protected:

                    AbstractDomain(gvt::core::math::AffineTransformMatrix<float> m = gvt::core::math::AffineTransformMatrix<float>(true));
                    AbstractDomain(const AbstractDomain &other);
                    virtual ~AbstractDomain();
                    
                public:

                    virtual bool intersect(gvt::render::actor::Ray&  r, gvt::render::actor::isecDomList& inter);
                    
                    virtual void marchIn(gvt::render::actor::Ray&  r);
                    virtual void marchOut(gvt::render::actor::Ray&  r);
                    virtual void trace(gvt::render::actor::RayVector& rayList, gvt::render::actor::RayVector& moved_rays);
                    

                    virtual bool load();
                    virtual void free();
                    virtual int size() = 0;
                    virtual int sizeInBytes() = 0;

                    virtual gvt::render::actor::Ray toLocal(gvt::render::actor::Ray& r);

                    virtual gvt::render::actor::Ray toWorld(gvt::render::actor::Ray& r);

                    virtual gvt::core::math::Vector4f toLocal(const gvt::core::math::Vector4f& r);

                    virtual gvt::core::math::Vector4f toWorld(const gvt::core::math::Vector4f& r);

                    virtual gvt::core::math::Vector4f localToWorldNormal(const gvt::core::math::Vector4f &v);
                    virtual gvt::render::data::primitives::Box3D getWorldBoundingBox();

                    virtual void setBoundingBox(gvt::render::data::primitives::Box3D bb);
                    
                    virtual gvt::render::data::primitives::Box3D getBounds(int type) const;

                    virtual bool domainIsLoaded();

                    virtual int getDomainID();

                    virtual void setDomainID(int id);
                    
                    virtual void translate(gvt::core::math::Vector4f t);
                    virtual void rotate(gvt::core::math::Vector4f t);
                    virtual void scale(gvt::core::math::Vector4f t);

                    virtual gvt::core::math::Point4f worldCentroid() const;
                    
                    
        //            virtual pop(gvt::render::actor::RayVector &queue, gvt::render::actor::Ray& ray) {
        //                boost::mutex::scoped_lock lock(_inqueue);
        //                if(queue.empty()) return false;
        //                ray = queue.back();
        //                queue.pop_back();
        //                return true;
        //                
        //            }
        //                        
        //            virtual void push(gvt::render::actor::RayVector &queue, gvt::render::actor::Ray&  r) {
        //                boost::mutex::scoped_lock lock(_inqueue);
        //                queue.push_back(r);
        //            }
        //            
        //            virtual void dispatch(gvt::render::actor::RayVector &queue, gvt::render::actor::Ray&  r) {
        //                boost::lock_guard<boost::mutex> _lock(_outqueue);
        //                queue.push_back(r);
        //            }
                    // Public variables
                    gvt::core::math::AffineTransformMatrix<float> m;
                    gvt::core::math::AffineTransformMatrix<float> minv;
                    gvt::core::math::Matrix3f normi;
                    gvt::render::data::primitives::Box3D boundingBox;

                    boost::mutex _inqueue;
                    boost::mutex _outqueue;
                    
                    int domainID;

                    bool isLoaded;

                };

            }
        }
    }
}
#endif // GVT_RENDER_DATA_DOMAIN_H
