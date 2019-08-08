/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray tracing
   tacc.github.io/GraviT

   Copyright 2013-2018 Texas Advanced Computing Center, The University of Texas at Austin
   All rights reserved.

   Licensed under the BSD 3-Clause License, (the "License"); you may not use this file
   except in compliance with the License.
   A copy of the License is included with this software in the file LICENSE.
   If your copy does not contain the License, you may obtain a copy of the License at:

http://opensource.org/licenses/BSD-3-Clause

Unless required by applicable law or agreed to in writing, software distributed under
the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.
See the License for the specific language governing permissions and limitations under
limitations under the License.

GraviT is funded in part by the US National Science Foundation under awards ACI-1339863,
ACI-1339881 and ACI-1339840
======================================================================================= */
#define TBB_PREVIEW_STATIC_PARTITIONER 1
#include "PVolAdapter.h"

#include <Lighting.h>

using namespace gvt::render::adapter::pvol::data;

bool PVolAdapter::init = false;
::pvol::Application*   PVolAdapter::theApplication;
::pvol::RendererP PVolAdapter::theRenderer;

// constructor for data (not implemented)
//OSPRayAdapter::OSPRayAdapter(gvt::render::data::primitives::Data *data) : Adapter(data) {
//  theOSPRenderer = ospNewRenderer("ptracer");
//}
// constructor for mesh data (not implemented)
//OSPRayAdapter::OSPRayAdapter(gvt::render::data::primitives::Mesh *data) : Adapter(data) {
//  theOSPRenderer = ospNewRenderer("ptracer");
//}
/***
 * following the function of the other adapters all this one does is map the data
 * in the GVT volume to ospray datatypes. If we are doing this right then this is
 * the first place in an application where ospray calls are made. So this is a
 * reasonable place to init ospray. In the end you should have an initialized
 * ospray volume object. The adapter needs to maintain a pointer to an ospray model
 * object.
 */
PVolAdapter::PVolAdapter(std::shared_ptr<gvt::render::data::primitives::Data> d, int w, int h) : Adapter(d) {

  std::shared_ptr<gvt::render::data::primitives::Volume> data = std::dynamic_pointer_cast<gvt::render::data::primitives::Volume>(d);

  GVT_ASSERT(data,"Data received by PVol Adapter is not a volume");

  int n_slices, n_isovalues;
  glm::vec4 *slices;
  glm::vec3 globalorigin;
  glm::vec3 localoffset(0.,0.,0.);
  glm::vec3 volumedimensions;
  glm::vec3 volumespacing;
  float *isovalues;

  cntx::rcontext &db = cntx::rcontext::instance();

  width  = w;
  height = h; 
  //std::cerr << "width " << width << " height " << height << std::endl;
  // build the PVOL volume from the data in the GraviT volume
  //std::cerr << " pvoladapter: init local volume shared p " << std::endl;
  ::pvol::VolumeP theVolume = ::pvol::Volume::NewP();
  //std::cerr << " pvoladapter: init local dataset shared p " << std::endl;
  theDataset = ::pvol::Datasets::NewP();
  //std::cerr << " pvoladapter: init local volume vis shared p " << std::endl;
  ::pvol::VolumeVisP theVolumeVis = ::pvol::VolumeVis::NewP();
  //std::cerr << " pvoladapter: init theVisualization ivar " << std::endl;
  theVisualization = ::pvol::Visualization::NewP();
  dolights = false;

  data->GetSlices(n_slices, slices);
  if (n_slices > 0) {
    dolights = true;
    ::pvol::vec4f *slicevector = new ::pvol::vec4f[n_slices];
    for (int i = 0; i < n_slices; i++) 
      slicevector[i] = ::pvol::vec4f(glm::value_ptr(slices[i]));
    theVolumeVis->SetSlices(n_slices, slicevector);
  }

  data->GetIsovalues(n_isovalues, isovalues);
  if (n_isovalues > 0) {
    //std::cout << "got isovalue " << n_isovalues << " " << std::endl;
    dolights = true;
    theVolumeVis->SetIsovalues(n_isovalues, isovalues);
  }

  data->GetGlobalOrigin(globalorigin);
  //std::cerr << "pvoladapter: set volume global origin" << std::endl;
  theVolume->set_global_origin(globalorigin.x, globalorigin.y, globalorigin.z);
  //std::cerr << "pvoladapter: set local offsets" << std::endl;
  theVolume->set_local_offset(localoffset.x, localoffset.y, localoffset.z);
  //std::cerr << "pvoladapter: set ghosted local offsets" << std::endl;
  theVolume->set_ghosted_local_offset(localoffset.x, localoffset.y, localoffset.z);

  data->GetCounts(volumedimensions);
  //std::cerr << "pvoladapter: set volume counts" << std::endl;
  theVolume->set_global_counts(volumedimensions.x, volumedimensions.y, volumedimensions.z);
  //std::cerr << "pvoladapter: set local counts" << std::endl;
  theVolume->set_local_counts(volumedimensions.x, volumedimensions.y, volumedimensions.z);
  //std::cerr << "pvoladapter: set ghosted local counts" << std::endl;
  theVolume->set_ghosted_local_counts(volumedimensions.x, volumedimensions.y, volumedimensions.z);
  
  data->GetDeltas(volumespacing);
  //std::cerr << "pvoladapter: set volume spacing" << std::endl;
  theVolume->set_deltas(volumespacing.x, volumespacing.y, volumespacing.z);

  //std::cerr << "pvoladapter: set volume samples" << std::endl;
  theVolume->set_samples((void*)data->GetSamples()); // same for float and uchar
  gvt::render::data::primitives::Box3D *bbox = data->getBoundingBox();
  glm::vec3 lowerbound = bbox->bounds_min;
  glm::vec3 upperbound = bbox->bounds_max;
  ::pvol::vec3f lb = {lowerbound.x,lowerbound.y,lowerbound.z};
  ::pvol::vec3f ub = {upperbound.x,upperbound.y,upperbound.z};
  //std::cerr << " boxL \n" << lowerbound.x << " " << lowerbound.y << " " << lowerbound.z << std::endl;
  //std::cerr << " boxU \n" << upperbound.x << " " << upperbound.y << " " << upperbound.z << std::endl;
  
  theVolume->set_local_box(lb,ub);
  gvt::render::data::primitives::Volume::VoxelType vt = data->GetVoxelType();
  //std::cerr << "pvoladapter: set volume data type" << std::endl;
  switch (vt) {
  case gvt::render::data::primitives::Volume::FLOAT: 
    theVolume->set_type(::pvol::Volume::DataType::FLOAT);
    break;
  case gvt::render::data::primitives::Volume::UCHAR: 
    theVolume->set_type(::pvol::Volume::DataType::UCHAR);
    break;
  default:
    std::cerr << "unrecognized voxel type in PVOL adapter: " << vt << std::endl;
  }
  // PVOL does not expose sampling rate
  //ospSet1f(theOSPVolume, "samplingRate",  data->GetSamplingRate());
  // PVOL color and opacity are exposed as ranged sets
  // (value, r,g,b) and (value, opacity)
  // so get the gvt data min-max and scale appropriately
  data->GetTransferFunction()->set();
  glm::vec2 datavaluerange;
  float data_min, data_max;
  //std::cerr << "pvoladapter commit the volume" << std::endl;
  theVolume->Commit();
  //std::cerr << "pvoladapter insert volume into dataset" << std::endl;
  theDataset->Insert("avol",theVolume);
  //std::cerr << "pvoladapter commit the dataset" << std::endl;
  theDataset->Commit();
  datavaluerange = data->GetTransferFunction()->getValueRange();
  data_min = datavaluerange[0]; data_max = datavaluerange[1];
  //theVolume->get_global_minmax(data_min, data_max);
  //std::cerr << "pvoladapter data min/max " << data_min << " " << data_max << std::endl;
  {
    int count = data->GetTransferFunction()->getColorCount();
    //std::cerr << " setting " << count << " element transfer function " << std::endl;
    // this statement is wrong. it gets the wrong vector
    // the right vector is a vec4 also. 
    //glm::vec3 *in = data->GetTransferFunction()->getColors();
    glm::vec4 *in = data->GetTransferFunction()->getColorMap();
    // copy it to a pvol vec4 stupid extra crap
    ::pvol::vec4f *out = new ::pvol::vec4f[count];
    float data_step = (data_max - data_min) / (float)(count+1);
    float scalar = data_min;
    for (int i = 0; i < count; ++i )
    {
      scalar = in[i].x*(data_max - data_min) + data_min;
      //std::cerr << i << "incolor " << in[i].x << " " << in[i].y <<  " " << in[i].z << " " << in[i].w << std::endl;
      out[i] = ::pvol::vec4f(scalar, in[i].y, in[i].z, in[i].w);
      //std::cerr << i << " color " << out[i].x << " " << out[i].y << " " << out[i].z << " " << out[i].w << std::endl;
      //scalar += data_step;
    }
    theVolumeVis->SetColorMap( count, out );
    delete[] out;
  }
  {
    int count = data->GetTransferFunction()->getOpacityCount();
    //float *in = data->GetTransferFunction()->getOpacity();
    glm::vec2 *in = data->GetTransferFunction()->getOpacityMap();
    ::pvol::vec2f *out = new ::pvol::vec2f[count];
    float data_step = (data_max - data_min) / (float)(count+1);
    float scalar = data_min;
    for (int i = 0; i < count; ++i )
    {
      scalar = in[i].x*(data_max - data_min) + data_min;
      out[i] = ::pvol::vec2f(scalar,in[i].y);
      std::cout << i << " opacity " << out[i].x << " " << out[i].y << std::endl;
      //scalar += data_step;
    }
    theVolumeVis->SetOpacityMap( count, out );
    delete[] out;
  }
  theVolumeVis->SetVolumeRendering( true );

  theVolumeVis->SetTheData(theVolume);
  theVolumeVis->SetName("avol");
  //std::cerr << "pvoladapter: commit theVolumevis " << std::endl;
  theVolumeVis->Commit(theDataset);
  //std::cerr << "pvoladapter: add volumevis to fisualization" << std::endl;
  theVisualization->AddVolumeVis(theVolumeVis);
  //std::cerr << "pvoladapter: commit the visualization" << std::endl;
  theVisualization->Commit(theDataset);

  // only need this if PVOL handles explicit meshes
  // theOSPModel = ospNewModel();
  // ospAddVolume(theOSPModel, theOSPVolume);
  // ospCommit(theOSPModel);
  // the model should be added to the renderer
  // ospSetObject(theOSPRenderer, "model", theOSPModel);
  // ospCommit(theOSPRenderer);
}

/*** this routine maps pvol rays to gravit rays
 *
 */
void PVolAdapter::PVOL2GVTMoved_Rays( ::pvol::RayList *theOutRays, ::pvol::RayList &theInRays, gvt::render::actor::RayVector &moved_rays) {
    int outCount, inCount;
    //std::cerr << (theOutRays==NULL ? " Null Outrays " : " ") <<std::endl;
  if(theOutRays && (theOutRays->GetRayCount() != 0)) {
   outCount = theOutRays->GetRayCount();
   //std::cerr << " pvol2moved: outrays " << outCount << std::endl;
  } else {
      outCount = 0;
  }
  if(&theInRays && theInRays.GetRayCount() != 0) {
   inCount  = theInRays.GetRayCount();
   //std::cerr << " pvol2moved: inrays " << inCount << std::endl;
  } else {
      inCount =0;
  }
  moved_rays.resize(outCount + inCount);
  //std::cerr << "pvol2moved: resize moved_rays " << outCount + inCount << std::endl;
  // plug in the rays into moved_rays
  // the idea is to pile all the rays to moved_rays and let the scheduler sort 'em
  // first check the out rays. out consists of generated rays (ao, shadow, ?)
  for (int i = 0; i < outCount; i++) {
    //  std::cerr << "red " << theOutRays->get_r(i) 
     //           << " green " << theOutRays->get_g(i)
      //          << " blue " << theOutRays->get_b(i)
       //         << " opac " << theOutRays->get_o(i) << std::endl;
    gvt::render::actor::Ray &ray = moved_rays[i];
    ray.mice.origin.x    = theOutRays->get_ox(i);
    ray.mice.origin.y    = theOutRays->get_oy(i);
    ray.mice.origin.z    = theOutRays->get_oz(i);
    ray.mice.direction.x = theOutRays->get_dx(i);
    ray.mice.direction.y = theOutRays->get_dy(i);
    ray.mice.direction.z = theOutRays->get_dz(i);
    ray.mice.color.r     = theOutRays->get_r(i);
    ray.mice.color.g     = theOutRays->get_g(i);
    ray.mice.color.b     = theOutRays->get_b(i);
    ray.mice.w           = theOutRays->get_o(i); // store pvol opacity in the w component of the gvt ray
    ray.mice.t           = theOutRays->get_t(i);
    ray.mice.t_max       = theOutRays->get_tMax(i);
    ray.mice.id          = theOutRays->get_y(i) * width + theOutRays->get_x(i);
    ray.mice.type        = theOutRays->get_type(i); // TODO: gvt and pvol happen to define these the same way. BRITTLE!
    ray.mice.depth       = theOutRays->get_term(i); // TODO: gvt and pvol happen to define these the same way. BRITTLE!
  }

  // now do the inRays rays which may be terminated as indicated in their term variable.
  for (int i = 0; i < inCount; i++) {
//      std::cerr << "red " << theInRays.get_r(i) 
 //               << " green " << theInRays.get_g(i)
  //              << " blue " << theInRays.get_b(i)
   //             << " opac " << theInRays.get_o(i) << std::endl;
    gvt::render::actor::Ray &ray = moved_rays[i + outCount]; // bump to end of outRays block
    ray.mice.origin.x    = theInRays.get_ox(i);
    ray.mice.origin.y    = theInRays.get_oy(i);
    ray.mice.origin.z    = theInRays.get_oz(i);
    ray.mice.direction.x = theInRays.get_dx(i);
    ray.mice.direction.y = theInRays.get_dy(i);
    ray.mice.direction.z = theInRays.get_dz(i);
    ray.mice.color.r     = theInRays.get_r(i);
    ray.mice.color.g     = theInRays.get_g(i);
    ray.mice.color.b     = theInRays.get_b(i);
    ray.mice.w           = theInRays.get_o(i); // store pvol opacity in the w component of the gvt ray
    ray.mice.t           = theInRays.get_t(i);
    ray.mice.t_max       = theInRays.get_tMax(i);
    ray.mice.id          = theInRays.get_y(i) * width + theInRays.get_x(i);
    ray.mice.type        = theInRays.get_type(i); // TODO: gvt and pvol happen to define these the same way. BRITTLE!
    ray.mice.depth       = theInRays.get_term(i); // TODO: gvt and pvol happen to define these the same way. BRITTLE!
  }
}
// convert gravit to ospray rays format
::pvol::RayList PVolAdapter::GVT2PVOLRays(gvt::render::actor::RayVector &rayList) {
  const int rayCount = rayList.size();
  //std::cerr << "converting " << rayCount << " rays" << std::endl;
  ::pvol::RayList out(rayCount);
  for (int i = 0; i < rayCount; i++) {
    out.set_ox(i, rayList[i].mice.origin.x);
    out.set_oy(i, rayList[i].mice.origin.y);
    out.set_oz(i, rayList[i].mice.origin.z);
    out.set_dx(i, rayList[i].mice.direction.x);
    out.set_dy(i, rayList[i].mice.direction.y);
    out.set_dz(i, rayList[i].mice.direction.z);
    out.set_r (i, rayList[i].mice.color.r);
    out.set_g (i, rayList[i].mice.color.g);
    out.set_b (i, rayList[i].mice.color.b);
    out.set_o (i, rayList[i].mice.w); // volume renderer uses w to carry opacity in and out.
    out.set_t (i, 0.0); // from PVOL's perspective, this ray is just beginning
    out.set_tMax(i, rayList[i].mice.t_max);
    out.set_type(i, rayList[i].mice.type);  // TODO: gvt and pvol happen to define these the same way. BRITTLE!
    out.set_term(i, rayList[i].mice.depth); // TODO: gvt and pvol happen to define these the same way. BRITTLE!
    
    // x and y are calculated from ray id and image dimensions.
    out.set_x(i, rayList[i].mice.id % width);
    out.set_y(i, rayList[i].mice.id / width);
  }
  return out;
}

// this is the trace function that gets called by the scheduler to actually
// trace the rays. The signature is the same as for ospray as for the other
// engines. Lighting is not used for volume rendering unless implicit
// surfaces and/or slices are used. Still, a light vector is passed. It
// may be empty.
void PVolAdapter::trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays,
                        glm::mat4 *m, glm::mat4 *minv, glm::mat3 *normi,
                        gvt::core::Vector<std::shared_ptr<gvt::render::data::scene::Light> > &lights, 
                        size_t _begin,
                        size_t _end)
{
  // lights
  // todo sort point and area lights. For now assume point light.
  // gravit stores light position and color. ospray uses direction instead.
  // need to  derive direction from position. Assume all point lights
  // have direction pointing to origin. Also scale to unit vector.
  float *light_array  = new float[3 * lights.size()];
  float *light_arrayP = light_array;
  // if the adapter constructor has not created implicit surfaces or
  // slices for a volume data then dolignts will be false and we
  // dont need to deal with them. If however there is some geometry
  // then dolights will be true and we will process the lights.
  if (dolights) {
//    gvt::render::data::scene::Light lgt;
    for (std::shared_ptr<gvt::render::data::scene::Light> light : lights) {
      glm::vec3 pos = light->position;
      float d = 1 / sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
      light_arrayP[0] = -pos[0] * d;
      light_arrayP[1] = -pos[1] * d;
      light_arrayP[2] = -pos[2] * d;
      light_arrayP += 3;
    }
    theLighting.SetLights(lights.size(), light_array);
    // some light and effect control variables. These should be stashed in the
    // context rather than hardcoded here.
    float ka = 0.4f, kd = 0.6f, AO_radius = 0.f;
    int AO_rays = 0;
    bool do_shadows = false;
    theLighting.SetK( ka, kd );
    theLighting.SetAO( AO_rays, AO_radius );
    theLighting.SetShadowFlag( do_shadows );
  }
  // convert GVT RayVector into the OSPExternalRays used by ospray.
//  std::cout << " ospadapter GVT -> osprays " << std::endl;
  ::pvol::RayList theInRays = GVT2PVOLRays(rayList);
  // trace'em
//  std::cout << " ospadapter: ospTraceRays " << std::endl;
  ::pvol::TraceRays tracer;
  //std::cerr << " pvoladapter tracing " << theInRays.GetRayCount() << " rays " <<std::endl;
  //std::cerr << "pvoladapter visualization ispc pointer " << theVisualization->GetISPC() << std::endl;
  //::pvol::RayList& theOutRays = *(tracer.Trace(theLighting, theVisualization, &theInRays));
  //theInRays.print();
  ::pvol::RayList  *theOutRays = (tracer.Trace(theLighting, theVisualization, &theInRays));
  //theInRays.print();
  // push everything from out and rl into moved_rays for sorting into houses
  //std::cerr << " whats up with theOutRays " << (theOutRays == NULL) << std::endl;
  //std::cerr <<  theInRays.GetRayCount() << " input rays " << std::endl;

  PVOL2GVTMoved_Rays(theOutRays, theInRays, moved_rays);
  // out and rl are no longer needed since they have been copied into moved_rays so
  // whack 'em.
  // delete out;
  // delete rl;
}
void PVolAdapter::init_pvol(int *argc, char **argv) {
  if (!PVolAdapter::init) {
    ospInit(argc, (const char **)argv);
    PVolAdapter::theApplication = new ::pvol::Application( argc, &argv );
    PVolAdapter::theApplication->Start(false /* == no MPI */);
    ::pvol::Renderer::Initialize();
    PVolAdapter::theApplication->Run();
    PVolAdapter::theRenderer = ::pvol::Renderer::NewP();
    PVolAdapter::theRenderer->Commit();
    PVolAdapter::init = true;
  } else {
    GVT_WARNING_BACKTRACE(0,"PVolAdapter::init_pvol called when already initialized");
  }
}
PVolAdapter::~PVolAdapter() 
{ 
  delete PVolAdapter::theApplication;
  PVolAdapter::init = false;
}
