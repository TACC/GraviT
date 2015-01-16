/*
 * IntersectQueue.cpp
 *
 *  Created on: Nov 27, 2013
 *      Author: jbarbosa
 */

#include <GVT/Domain/domains.h>
#include <GVT/common/debug.h>
#include <GVT/Math/GVTMath.h>
//#include <Data/gvt_optix.h>
#include <GVT/Data/primitives.h>
#include "OptixProcessQueue.h"

#include <cuda_runtime.h>
#include <optix_prime/optix_primepp.h>
#include <optixu/optixpp.h>

// TODO: Matt
namespace GVT {

namespace Backend {

template <>
void ProcessQueue<GVT::Domain::OptixDomain>::IntersectDomain(
    GVT::Data::ray& ray, GVT::Data::RayVector& newRays) {}

template <>
void ProcessQueue<GVT::Domain::OptixDomain>::operator()() {
  cerr << "Entering ProcessQueue" << endl;
  GVT::Domain::GeometryDomain* gdom =
      dynamic_cast<GVT::Domain::GeometryDomain*>(param->dom);
  if (!gdom) return;
  GVT_DEBUG(DBG_ALWAYS, "processQueue<OptixDomain>: in\n");

  /* TODO :
   *  - Convert mesh to Optix Mesh or list of triangles (gdom->mesh)
   *  - Convert ray queue to Optix Rays (param->queue[param->domTarget];)
   */

  // Ray r={ make_float3(0,0,0), 0, make_float3(0,0,1), 1e34f };
  RTPcontexttype contextType = RTP_CONTEXT_TYPE_CUDA;
  optix::prime::Context context = optix::prime::Context::create(contextType);
  optix::prime::Model model = context->createModel();

  GVT::Data::RayVector& rayList = param->queue[param->domTarget];
  // copy rays to buffer
  optix::Ray* optixRays = new optix::Ray[rayList.size()];
  for (int i = 0; i < rayList.size(); i++) {
    optixRays[i].origin = make_float3(
        rayList[i].origin[0], rayList[i].origin[1], rayList[i].origin[2]);
    optixRays[i].direction =
        make_float3(rayList[i].direction[0], rayList[i].direction[1],
                    rayList[i].direction[2]);

    // TODO : [OPTIX] Gravity data structure does not have a tmin/tmax

    // optixRays[i].tmin     = rayList[i].tmin;
    // optixRays[i].tmax     = rayList[i].tmax;
  }

  // Buffer<Ray> rays( 0, bufferType, LOCKED );

  /* while(queue not empty)
   *      compute intersection
   *      if ray intersects
   *        generate next ray (path tracer) add to queue
   *        generate shadow rays add to queue
   *      else
   *        if shadow ray
   *           shade ( gdom->mesh->mat->shade )
   *        else if not shadow
   *           add ray to moved rays
   *
   **/

  delete[] optixRays;
}

}  // namespace Backend;

}  // namespace GVT
