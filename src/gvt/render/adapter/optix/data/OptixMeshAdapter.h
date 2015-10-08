#ifndef GVT_RENDER_ADAPTER_OPTIX_DATA_OPTIX_MESH_ADAPTER_H
#define GVT_RENDER_ADAPTER_OPTIX_DATA_OPTIX_MESH_ADAPTER_H

#include "gvt/render/Adapter.h"

#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_prime/optix_primepp.h>

namespace gvt {
namespace render {
namespace adapter {
namespace optix {
namespace data {

class OptixMeshAdapter : public gvt::render::Adapter
{
public:
    /**
     * Construct the Embree mesh adapter.  Convert the mesh
     * at the given node to Embree's format.
     *
     * Initializes Embree the first time it is called.
     */
    OptixMeshAdapter(gvt::core::DBNodeH node);

    /**
     * Release Embree copy of the mesh.
     */
    virtual ~OptixMeshAdapter();

    /**
     * Return the Embree scene handle;
     */
    ::optix::prime::Model getScene() const { return optix_model_; }

    /**
     * Return the geometry id.
     */
    unsigned getGeomId() const { return geomId; }

    /**
     * Return the packet size
     */
    unsigned long long getPacketSize () const { return packetSize; }

    /**
     * Trace rays using the Embree adapter.
     *
     * Creates threads and traces rays in packets defined by GVT_EMBREE_PACKET_SIZE
     * (currently set to 4).
     *
     * \param rayList incoming rays
     * \param moved_rays outgoing rays [rays that did not hit anything]
     * \param instNode instance db node containing dataRef and transforms
     */
    virtual void trace(gvt::render::actor::RayVector& rayList,
            gvt::render::actor::RayVector& moved_rays,
            gvt::core::DBNodeH instNode);

protected:


    /**
     * Handle to Optix context.
     */
    ::optix::prime::Context optix_context_;

    /**
     * Handle to Optix model.
     */
    ::optix::prime::Model optix_model_;


    float multiplier = 1.0f - 16.0f * std::numeric_limits<float>::epsilon(); 

    /**
     * Static bool to initialize Optix before use.
     * 
     * // TODO: this will need to move in the future when we have different types of Embree adapters (ex: mesh + volume)
     */
    static bool init;

    /**
     * Currently selected packet size flag.
     */
    unsigned long long packetSize;

    /**
     * Handle to Embree scene.
     */
    // RTCScene scene;

    /**
     * Handle to the Embree triangle mesh.
     */
    unsigned geomId;
};

}
}
}
}
}

#endif /*GVT_RENDER_ADAPTER_OPTIX_DATA_OPTIX_MESH_ADAPTER_H*/