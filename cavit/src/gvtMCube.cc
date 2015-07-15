// #include <Model/Primitives/Cube.h>
#include "gvtMCube.h"
#include <Interface/RayPacket.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>

#include <Model/Intersections/AxisAlignedBox.h>

using namespace Manta;
using namespace std;

gvtMCube::gvtMCube(Material* mat, const Vector& p0, const Vector& p1, unsigned int id, Instance* instance)
  : PrimitiveCommon(mat), domainId(id), instance(instance)
{
  bbox[0] = Min(p0, p1);
  bbox[1] = Max(p0, p1);
}

gvtMCube::~gvtMCube()
{
}

void gvtMCube::setMinMax(const Vector&  p0, const Vector& p1)
{
  bbox[0] = Min(p0, p1);
  bbox[1] = Max(p0, p1);
}

void gvtMCube::computeBounds(const PreprocessContext&, BBox& bbox_) const
{
  //  bbox.extendByPoint(Point(xmin, ymin, zmin));
  //  bbox.extendByPoint(Point(xmax, ymax, zmax));
  bbox_.extendByPoint( bbox[0] );
  bbox_.extendByPoint( bbox[1] );
}

void gvtMCube::intersect(const RenderContext& context, RayPacket& rays) const
{

  // This is so our t values have the same scale as other prims, e.g., TexTriangle
  rays.normalizeDirections();
  
  // Intersection algorithm requires inverse directions computed.
  rays.computeInverseDirections();
  rays.computeSigns();

  instance->intersect(context, rays);
  for (int i=rays.begin();i<rays.end();++i) {
    if (rays.wasHit(i))
      rays.getScratchpad<unsigned int>(0)[i] = domainId;
  }

  return;

  // Iterate over each ray.
  for (int i=rays.begin();i<rays.end();++i) {

    Real tmin, tmax;
    bool wasHit = false;
    // Check for an intersection.
    if (intersectAaBox( bbox,
                                      tmin,
                                      tmax,
                                      rays.getRay(i),
                                      rays.getSigns(i),
                                      rays.getInverseDirection(i))){
          // Check to see if we are inside the box.
      if (tmin > T_EPSILON)
      {
        if (rays.hit( i, tmin, getMaterial(), this, getTexCoordMapper() ))
          wasHit=true;
      }
      // And use the max intersection if we are.
      else
      {
        if (rays.hit( i, tmax, getMaterial(), this, getTexCoordMapper() ))
          wasHit=true;
      }
      if (wasHit)
        rays.getScratchpad<unsigned int>(0)[i] = domainId;
    }
  }
}


void gvtMCube::computeNormal(const RenderContext&, RayPacket& rays) const
{
  rays.computeHitPositions();
  for(int i=rays.begin(); i<rays.end(); i++) {
    Vector hp = rays.getHitPosition(i);
    if (Abs(hp.x() - bbox[0][0]) < 0.0001)
      rays.setNormal(i, Vector(-1, 0, 0 ));
    
    else if (Abs(hp.x() - bbox[1][0]) < 0.0001)
      rays.setNormal(i, Vector( 1, 0, 0 ));
    
    else if (Abs(hp.y() - bbox[0][1]) < 0.0001)
      rays.setNormal(i, Vector( 0,-1, 0 ));
    
    else if (Abs(hp.y() - bbox[1][1]) < 0.0001)
      rays.setNormal(i, Vector( 0, 1, 0 ));
    
    else if (Abs(hp.z() - bbox[0][2]) < 0.0001)
      rays.setNormal(i, Vector( 0, 0,-1 ));
    
    else 
      rays.setNormal(i, Vector( 0, 0, 1 ));
  }
}
