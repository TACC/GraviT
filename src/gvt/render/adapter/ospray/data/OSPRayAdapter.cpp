/* ======================================================================================= 
   This file is released as part of GraviT - scalable, platform independent ray tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas at Austin  
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
#include "gvt/render/adapter/ospray/data/OSPRayAdapter.h"
#include "gvt/core/CoreContext.h"

using namespace gvt::render::adapter::ospray::data;

bool OSPRayAdapter::init = false;

// constructor for data (not implemented)
OSPRayAdapter::OSPRayAdapter(gvt::render::data::primitives::Data *data):Adapter(data) {
    theOSPRenderer = ospNewRenderer("ptracer");
}
// constructor for mesh data (not implemented)
OSPRayAdapter::OSPRayAdapter(gvt::render::data::primitives::Mesh *data):Adapter(data) {
    theOSPRenderer = ospNewRenderer("ptracer");
}
/***
 * following the function of the other adapters all this one does is map the data
 * in the GVT volume to ospray datatypes. If we are doing this right then this is
 * the first place in an application where ospray calls are made. So this is a 
 * reasonable place to init ospray. In the end you should have an initialized
 * ospray volume object. The adapter needs to maintain a pointer to an ospray model 
 * object.
 */
OSPRayAdapter::OSPRayAdapter(gvt::render::data::primitives::Volume *data):Adapter(data) {
    int n_slices,n_isovalues;
    glm::vec4 *slices;
    glm::vec3 globalorigin;
    glm::vec3 volumedimensions;
    glm::vec3 volumespacing;
    float *isovalues;
    gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
    gvt::core::DBNodeH root = cntxt->getRootNode();
    width = root["Film"]["width"].value().toInteger();
    height = root["Film"]["height"].value().toInteger();
    theOSPRenderer = ospNewRenderer("ptracer");
    // build the ospray volume from the data in the GraviT volume
    theOSPVolume = ospNewVolume("shared_structured_volume");
    dolights = false;
    data->GetSlices(n_slices,slices);
    if(n_slices != 0) {
        dolights = true;
        float *slicevector = new float[4*n_slices];
        for(int i=0;i<n_slices;i++)
            std::memcpy(glm::value_ptr(slices[i]),slicevector+(4*i),4*sizeof(float));
        OSPData sliceData = ospNewData(n_slices,OSP_FLOAT4,slicevector);
        ospCommit(sliceData);
        ospSetData(theOSPVolume,"slices",sliceData);
    }
    data->GetIsovalues(n_isovalues,isovalues);
    if(n_isovalues != 0) {
        std::cout << "got isovalue " << n_isovalues << " "  << std::endl;
        dolights = true;
        OSPData isoData = ospNewData(n_isovalues,OSP_FLOAT,isovalues);
        ospCommit(isoData);
        ospSetData(theOSPVolume,"isovalues",isoData);
    }
    data->GetGlobalOrigin(globalorigin);
    osp::vec3f origin;
    origin.x = globalorigin.x;
    origin.y = globalorigin.y;
    origin.z = globalorigin.z;
    ospSetVec3f(theOSPVolume,"gridOrigin",origin);
    data->GetCounts(volumedimensions);
    osp::vec3i counts;
    counts.x = volumedimensions.x;
    counts.y = volumedimensions.y;
    counts.z = volumedimensions.z;
    ospSetVec3i(theOSPVolume,"dimensions",counts);
    data->GetDeltas(volumespacing);
    osp::vec3f spacing;
    spacing.x = volumespacing.x;
    spacing.y = volumespacing.y;
    spacing.z = volumespacing.z;
    ospSetVec3f(theOSPVolume,"gridSpacing",spacing);
    gvt::render::data::primitives::Volume::VoxelType vt = data->GetVoxelType();
    // as of now only two voxel types are supported by the ospray lib
    switch(vt){
        case gvt::render::data::primitives::Volume::FLOAT : ospSetString(theOSPVolume,"voxelType","float");
                                                            int numberofsamples = counts.x*counts.y*counts.z;
                                                            OSPData voldata = ospNewData(numberofsamples,OSP_FLOAT,(void*)data->GetSamples(),OSP_DATA_SHARED_BUFFER);
                                                            ospCommit(voldata);
                                                            ospSetObject(theOSPVolume,"voxelData",voldata);
                                                            break;
        case gvt::render::data::primitives::Volume::UCHAR : ospSetString(theOSPVolume,"voxelType","uchar");
                                                            break;
        default : std::cerr << " error setting voxel type " << std::endl;
                  break;
    }
    ospSet1f(theOSPVolume,"samplingRate",data->GetSamplingRate());
    data->GetTransferFunction()->set();
    ospSetObject(theOSPVolume,"transferFunction",data->GetTransferFunction()->GetTheOSPTransferFunction());
    ospSet1i(theOSPVolume,"volume rendering",1);
    ospCommit(theOSPVolume);
    // make a model and stuff the volume in it.
    theOSPModel = ospNewModel();
    ospAddVolume(theOSPModel,theOSPVolume);
    ospCommit(theOSPModel);
    // the model should be added to the renderer
    ospSetObject(theOSPRenderer,"model",theOSPModel);
    ospCommit(theOSPRenderer);
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
        moved_rays.resize(raycount);
        for (int i=0; i< out->GetCount(); i++) {
            gvt::render::actor::Ray &ray = moved_rays[i];
            ray.origin.x = out->xr.ox[i];
            ray.origin.y = out->xr.oy[i];
            ray.origin.z = out->xr.oz[i];
            ray.direction.x = out->xr.dx[i];
            ray.direction.y = out->xr.dy[i];
            ray.direction.z = out->xr.dz[i];
            ray.color.r = out->xr.r[i];
            ray.color.g = out->xr.g[i];
            ray.color.b = out->xr.b[i];
            ray.w = out->xr.o[i]; // store ospray opacity in the w component of the gvt ray
            ray.t = out->xr.t[i];
            ray.t_max = out->xr.tMax[i];
            ray.id = out->xr.y[i]*width + out->xr.x[i];
            ray.type = out->xr.type[i] == EXTERNAL_RAY_PRIMARY ? RAY_PRIMARY :
                out->xr.type[i] == EXTERNAL_RAY_SHADOW ? RAY_SHADOW :
                out->xr.type[i] == EXTERNAL_RAY_AO ? RAY_AO : RAY_EMPTY;
            ray.depth = (out->xr.term[i] & EXTERNAL_RAY_SURFACE ? RAY_SURFACE : 0 ) |
                (out->xr.term[i] & EXTERNAL_RAY_OPAQUE ? RAY_OPAQUE : 0) |
                (out->xr.term[i] & EXTERNAL_RAY_BOUNDARY ? RAY_BOUNDARY : 0) |
                (out->xr.term[i] & EXTERNAL_RAY_TIMEOUT ? RAY_TIMEOUT : 0);
        }
    } else { raycount = 0; }
    // now do the rl rays which may be terminated as indicated in their term variable.  
    moved_rays.resize(raycount + rl->GetCount());
    for(int i=raycount, ii=0; i <raycount + rl->GetCount();i++,ii++){
        gvt::render::actor::Ray &ray = moved_rays[i];
        ray.origin.x = rl->xr.ox[ii];
        ray.origin.y = rl->xr.oy[ii];
        ray.origin.z = rl->xr.oz[ii];
        ray.direction.x = rl->xr.dx[ii];
        ray.direction.y = rl->xr.dy[ii];
        ray.direction.z = rl->xr.dz[ii];
        ray.color.r = rl->xr.r[ii];
        ray.color.g = rl->xr.g[ii];
        ray.color.b = rl->xr.b[ii];
        ray.w = rl->xr.o[ii];
        ray.t = rl->xr.t[ii];
        ray.t_max = rl->xr.tMax[ii];
        ray.id = rl->xr.y[ii]*width + rl->xr.x[ii];
        ray.type = rl->xr.type[ii] == EXTERNAL_RAY_PRIMARY ? RAY_PRIMARY :
            rl->xr.type[ii] == EXTERNAL_RAY_SHADOW ? RAY_SHADOW :
            rl->xr.type[ii] == EXTERNAL_RAY_AO ? RAY_AO : RAY_EMPTY;
        ray.depth = (rl->xr.term[ii] & EXTERNAL_RAY_SURFACE ? RAY_SURFACE : 0 ) |
            (rl->xr.term[ii] & EXTERNAL_RAY_OPAQUE ? RAY_OPAQUE : 0) |
            (rl->xr.term[ii] & EXTERNAL_RAY_BOUNDARY ? RAY_BOUNDARY : 0) |
            (rl->xr.term[ii] & EXTERNAL_RAY_TIMEOUT ? RAY_TIMEOUT : 0);
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
        //out->xr.t[i] = rayList[i].t;
        out->xr.t[i] = 0.0;
        out->xr.tMax[i] = rayList[i].t_max;
        out->xr.type[i] = rayList[i].type == RAY_PRIMARY ? EXTERNAL_RAY_PRIMARY :
            rayList[i].type == RAY_SHADOW ? EXTERNAL_RAY_SHADOW :
            rayList[i].type == RAY_AO ? EXTERNAL_RAY_AO : EXTERNAL_RAY_EMPTY;
        out->xr.term[i] = rayList[i].depth;
        // x and y are calculated from ray id and image dimensions. 
        out->xr.x[i] = rayList[i].id % width; 
        out->xr.y[i] = rayList[i].id / width; 
    }
    return out;
}

// this is the trace function that gets called by the scheduler to actually
// trace the rays. The signature is the same as for ospray as for the other
// engines. Lighting is not used for volume rendering unless implicit 
// surfaces and/or slices are used. Still, a light vector is passed. It
// may be empty. 
void OSPRayAdapter::trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays, glm::mat4 *m, glm::mat4 *minv, glm::mat3 *normi, std::vector<gvt::render::data::scene::Light *> &lights, size_t begin ,size_t end) { 
    // lights
    // todo sort point and area lights. For now assume point light. 
    // gravit stores light position and color. ospray uses direction instead. 
    // need to  derive direction from position. Assume all point lights 
    // have direction pointing to origin. Also scale to unit vector.
    float* lghts = new float[3*lights.size()];
    float* lghtptr = lghts;
    // if the adapter constructor has not created implicit surfaces or
    // slices for a volume data then dolignts will be false and we
    // dont need to deal with them. If however there is some geometry
    // then dolights will be true and we will process the lights. 
    if(dolights) {
        gvt::render::data::scene::Light lgt;
        for(gvt::render::data::scene::Light *lgt : lights) {
            glm::vec3 pos = lgt->position;
            float d = 1/sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]);
            lghtptr[0] = -pos[0]*d;
            lghtptr[1] = -pos[1]*d;
            lghtptr[2] = -pos[2]*d;
            lghtptr +=3;
        }
        OSPData lightData = ospNewData(lights.size(),OSP_FLOAT3,lghts);
        ospSetData(theOSPRenderer,"lights",lightData);
        // some light and effect control variables. These should be stashed in the
        // context rather than hardcoded here. 
        float ka,kd,arad;
        int nar;
        bool dos;
        // these are only used in the case of isosurface or slice planes I think.
        ka = 0.4;
        kd = 0.6;
        // do shadows boolean. Not used since I need an integer.
        dos = false;
        // ambient occlusion radius 
        arad = 0.0;
        // number of ambient occlusion rays. 
        nar = 0;
        ospSet1i(theOSPRenderer,"do_shadows",0);
        ospSet1i(theOSPRenderer,"n_ao_rays",nar);
        ospSet1f(theOSPRenderer,"ao_radius",arad);
        ospSet1f(theOSPRenderer,"Ka",ka);
        ospSet1f(theOSPRenderer,"Kd",kd);
        ospCommit(theOSPRenderer);
    }
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

