#define TBB_PREVIEW_STATIC_PARTITIONER 1
#include "gvt/render/adapter/ospray/data/OSPRayAdapter.h"
#include "gvt/core/CoreContext.h"

using namespace gvt::render::adapter::ospray::data;

bool OSPRayAdapter::init = false;

OSPRayAdapter::OSPRayAdapter(gvt::render::data::primitives::Data *data):Adapter(data) {
  theOSPRenderer = ospNewRenderer("ptracer");
}
OSPRayAdapter::OSPRayAdapter(gvt::render::data::primitives::Mesh *data):Adapter(data) {
  theOSPRenderer = ospNewRenderer("ptracer");
}
OSPRayAdapter::OSPRayAdapter(gvt::render::data::primitives::Volume *data):Adapter(data) {
  int n_slices,n_isovalues;
  glm::vec4 *slices;
  glm::vec3 globalorigin;
  glm::vec3 volumedimensions;
  glm::vec3 volumespacing;
  float *isovalues;

  theOSPRenderer = ospNewRenderer("ptracer");
  // build the ospray volume from the gvt volume
  if(theOSPVolume) ospRelease(theOSPVolume);
  if(theOSPData) ospRelease(theOSPData);
  theOSPVolume = ospNewVolume("shared_structured_volume");
  data->GetSlices(n_slices,slices);
  if(n_slices != 0) {
    float *slicevector = new float[4*n_slices];
    for(int i=0;i<n_slices;i++)
      std::memcpy(glm::value_ptr(slices[i]),slicevector+(4*i),4*sizeof(float));
      //slicevector[i*4] = glm::value_ptr(slices[i]);
      OSPData sliceData = ospNewData(n_slices,OSP_FLOAT4,slicevector);
      ospSetData(theOSPVolume,"slices",sliceData);
  }
  data->GetIsovalues(n_isovalues,isovalues);
  if(n_isovalues != 0) {
    OSPData isoData = ospNewData(n_isovalues,OSP_FLOAT,isovalues);
    ospSetData(theOSPVolume,"isovalues",isoData);
  }
  data->GetGlobalOrigin(globalorigin);
  osp::vec3f origin;
  origin.x = globalorigin.x;
  origin.y = globalorigin.y;
  origin.z = globalorigin.z;
  ospSetVec3f(theOSPVolume,"gridOrigin",origin);
  data->GetCounts(volumedimensions);
  osp::vec3f counts;
  counts.x = volumedimensions.x;
  counts.y = volumedimensions.y;
  counts.z = volumedimensions.z;
  ospSetVec3f(theOSPVolume,"dimensions",counts);
  data->GetDeltas(volumespacing);
  osp::vec3f spacing;
  spacing.x = volumespacing.x;
  spacing.y = volumespacing.y;
  spacing.z = volumespacing.z;
  ospSetVec3f(theOSPVolume,"gridSpacing",spacing);

}

/*** this routine maps ospexternal rays to gravit rays
 *
 */
void OSPRayAdapter::OSP2GVTMoved_Rays(OSPExternalRays &out, OSPExternalRays &rl, gvt::render::actor::RayVector &moved_rays) {
  int raycount;
  // plug in the rays into moved_rays
  // the idea is to pile all the rays to moved_rays and let the scheduler sort 'em
  // first check the out rays. out consists of generated rays (ao, shadow, ?) 
  if( out && out->GetCount() != 0) { // pack the output into moved_rays
    raycount = out->GetCount();
    for (int i=0; i< out->GetCount(); i++) {
      moved_rays[i].origin.x = out->xr.ox[i];
      moved_rays[i].origin.y = out->xr.oy[i];
      moved_rays[i].origin.z = out->xr.oz[i];
      moved_rays[i].direction.x = out->xr.dx[i];
      moved_rays[i].direction.y = out->xr.dy[i];
      moved_rays[i].direction.z = out->xr.dz[i];
      moved_rays[i].color.r = out->xr.r[i];
      moved_rays[i].color.g = out->xr.g[i];
      moved_rays[i].color.b = out->xr.b[i];
      moved_rays[i].w = out->xr.o[i];
      moved_rays[i].t = out->xr.t[i];
      moved_rays[i].t_max = out->xr.tMax[i];
      moved_rays[i].id = out->xr.x[i];
      moved_rays[i].depth = out->xr.y[i];
      moved_rays[i].type = out->xr.type[i] == EXTERNAL_RAY_PRIMARY ? RAY_PRIMARY :
        out->xr.type[i] == EXTERNAL_RAY_SHADOW ? RAY_SHADOW :
        out->xr.type[i] == EXTERNAL_RAY_AO ? RAY_AO : RAY_EMPTY;
      // note to future dave. Here we build a bit pattern that indicates the
      // termination type of the ray. We then store it in the t_min member of 
      // the ith moved ray. This implies special handling of the moved_rays 
      // in shuffle rays for volume rendering. 
      int spoot = (out->xr.term[i] & EXTERNAL_RAY_SURFACE ? RAY_SURFACE : 0 ) |
        (out->xr.term[i] & EXTERNAL_RAY_OPAQUE ? RAY_OPAQUE : 0) |
        (out->xr.term[i] & EXTERNAL_RAY_BOUNDARY ? RAY_BOUNDARY : 0) |
        (out->xr.term[i] & EXTERNAL_RAY_TIMEOUT ? RAY_TIMEOUT : 0);
      memcpy((void*)&(moved_rays[i].t_min), (void*) &spoot, sizeof(spoot));
    }
  } else { raycount = 0; }
  // now do the rl rays which may be terminated as indicated in their term variable.  
  for(int i=raycount; i <rl->GetCount();i++){
    moved_rays[i].origin.x = rl->xr.ox[i];
    moved_rays[i].origin.y = rl->xr.oy[i];
    moved_rays[i].origin.z = rl->xr.oz[i];
    moved_rays[i].direction.x = rl->xr.dx[i];
    moved_rays[i].direction.y = rl->xr.dy[i];
    moved_rays[i].direction.z = rl->xr.dz[i];
    moved_rays[i].color.r = rl->xr.r[i];
    moved_rays[i].color.g = rl->xr.g[i];
    moved_rays[i].color.b = rl->xr.b[i];
    moved_rays[i].w = rl->xr.o[i];
    moved_rays[i].t = rl->xr.t[i];
    moved_rays[i].t_max = rl->xr.tMax[i];
    moved_rays[i].id = rl->xr.x[i];
    moved_rays[i].depth = rl->xr.y[i];
    moved_rays[i].type = rl->xr.type[i] == EXTERNAL_RAY_PRIMARY ? RAY_PRIMARY :
      rl->xr.type[i] == EXTERNAL_RAY_SHADOW ? RAY_SHADOW :
      rl->xr.type[i] == EXTERNAL_RAY_AO ? RAY_AO : RAY_EMPTY;
    // note to future dave. Here we build a bit pattern that indicates the
    // termination type of the ray. We then store it in the t_min member of 
    // the ith moved ray. This implies special handling of the moved_rays 
    // in shuffle rays for volume rendering. 
    int spoot = (rl->xr.term[i] & EXTERNAL_RAY_SURFACE ? RAY_SURFACE : 0 ) |
      (rl->xr.term[i] & EXTERNAL_RAY_OPAQUE ? RAY_OPAQUE : 0) |
      (rl->xr.term[i] & EXTERNAL_RAY_BOUNDARY ? RAY_BOUNDARY : 0) |
      (rl->xr.term[i] & EXTERNAL_RAY_TIMEOUT ? RAY_TIMEOUT : 0);
    memcpy((void*)&(moved_rays[i].t_min), (void*) &spoot, sizeof(spoot));
  }
}
OSPExternalRays OSPRayAdapter::GVT2OSPRays(gvt::render::actor::RayVector &rayList) { 
  OSPExternalRays out = ospNewExternalRays() ;
  out->Allocate(rayList.size());
  for (int i = 0; i<rayList.size();i++) {
    out->xr.ox[i] = rayList[i].origin.x;
    out->xr.oy[i] = rayList[i].origin.y;
    out->xr.oz[i] = rayList[i].origin.z;
    out->xr.dx[i] = rayList[i].direction.x;
    out->xr.dy[i] = rayList[i].direction.y;
    out->xr.dz[i] = rayList[i].direction.z;
    out->xr.r[i] = rayList[i].color.r;
    out->xr.g[i] = rayList[i].color.g;
    out->xr.b[i] = rayList[i].color.b;
    out->xr.o[i] = rayList[i].w; // volume renderer uses w to carry opacity in and out.
    out->xr.t[i] = rayList[i].t;
    out->xr.tMax[i] = rayList[i].t_max;
    out->xr.type[i] == rayList[i].type == RAY_PRIMARY ? EXTERNAL_RAY_PRIMARY :
      rayList[i].type == RAY_SHADOW ? EXTERNAL_RAY_SHADOW :
      rayList[i].type == RAY_AO ? EXTERNAL_RAY_AO : EXTERNAL_RAY_EMPTY;
    out->xr.term[i] = 0;
    out->xr.x[i] = rayList[i].id; // volume renderer uses id to store px
    out->xr.y[i] = rayList[i].depth; // volume renderer uses depth to store py
  }
  return out;
}

void OSPRayAdapter::trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays, glm::mat4 *m, glm::mat4 *minv, glm::mat3 *normi, std::vector<gvt::render::data::scene::Light *> &lights, size_t begin ,size_t end) { 
  std::cout << " tracin" << std::endl; 
  // convert GVT RayVector into the OSPExternalRays used by ospray. 
  OSPExternalRays rl = GVT2OSPRays(rayList);
  // trace'em 
  OSPExternalRays out = ospTraceRays(theOSPRenderer,rl); // ospray trace
  // push everything from out and rl into moved_rays for sorting into houses
  // YA Griffindor. 
  OSP2GVTMoved_Rays(out,rl,moved_rays);
  // out and rl are no longer needed since they have been copied into moved_rays so 
  // whack 'em. 
  delete out;
  delete rl;
}
void OSPRayAdapter::initospray(int * argc, char**argv) {
  if (!OSPRayAdapter::init) {
    ospInit(argc,(const char**)argv);
    OSPRayAdapter::init = true;
  }
}
OSPRayAdapter::~OSPRayAdapter() {
  ospRelease(theOSPRenderer);
}

