
#ifndef Manta_Model_gvtMCube_h
#define Manta_Model_gvtMCube_h

#include <Model/Primitives/PrimitiveCommon.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/BBox.h>
// #include <Model/Groups/DynBVH.h>
#include <Model/Instances/Instance.h>
#include <Model/Instances/InstanceRT.h>
#include <Model/Instances/InstanceT.h>

namespace Manta
{

  class gvtMCube : public PrimitiveCommon {
  public:
    gvtMCube(Material* mat, const Vector& min_, const Vector& max_, unsigned int id, Instance* as);
    ~gvtMCube();
    void setMinMax(const Vector&  p0, const Vector& p1);
    virtual void computeBounds(const PreprocessContext& context,
                               BBox& bbox) const;
    virtual void intersect(const RenderContext& context, RayPacket& rays) const ;
    virtual void computeNormal(const RenderContext& context, RayPacket &rays) const;    
    
  private:
    BBox bbox;
    unsigned int domainId;
    // DynBVH* as;
    Instance* instance;
  };
}

#endif
