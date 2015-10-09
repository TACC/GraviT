#ifndef GVT_RENDER_ADAPTER_H
#define GVT_RENDER_ADAPTER_H

#include "gvt/core/DatabaseNode.h"

#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/mutex.hpp>

namespace gvt {
namespace render {
/// base class for ray tracing engine adapters
/**  MantaMeshAdapter, EmbreeMeshAdapter, OptixMeshAdapter */
class Adapter {
protected:
    /**
     * Data node (ex: Mesh, Volume)
     */
    gvt::core::DBNodeH node;

public:
    /**
     * Construct an adapter with a given data node
     */
    Adapter(gvt::core::DBNodeH node)
        : node(node)
    {}

    /**
     * Destroy the adapter
     */
    virtual ~Adapter()
    {}

    /**
     * Trace rays using the adapter.
     *
     * \param rayList incoming rays
     * \param moved_rays outgoing rays [rays that did not hit anything]
     * \param instNode instance db node containing dataRef and transforms
     */
    virtual void trace(gvt::render::actor::RayVector& rayList,
            gvt::render::actor::RayVector& moved_rays,
            gvt::core::DBNodeH instNode) = 0;

    boost::mutex _inqueue;
    boost::mutex _outqueue;
};

} // render
} // gvt

#endif // GVT_RENDER_ADAPTER_H
