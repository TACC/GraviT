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

// API functions
#include <gvt/core/context/Variant.h>
#include <gvt/render/RenderContext.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/Types.h>
#include <gvt/render/data/Domains.h>

#include <tbb/task_scheduler_init.h>
#include <thread>

#ifdef GVT_RENDER_ADAPTER_EMBREE
#include <gvt/render/adapter/embree/EmbreeMeshAdapter.h>
#endif

#ifdef GVT_RENDER_ADAPTER_MANTA
#include <gvt/render/adapter/manta/MantaMeshAdapter.h>
#endif

#ifdef GVT_RENDER_ADAPTER_OPTIX
#include <gvt/render/adapter/optix/OptixMeshAdapter.h>
#endif

void gvtInit() {
    // init mpi... or not
    int initialized,rank;
    MPI_Initialized(&initialized);
    if(!initialized)
    {
        MPI_Init(NULL,NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    }
    // initialize the context and a few other things. 
    gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
    if (cntxt == NULL) // context creation failed
    {
        GVT_ERR_MESSAGE("gvtInit: context init failed");
        exit(0);
    }
    else // build base context objects. 
    {
        // some data nodes
        gvt::core::DBNodeH root = cntxt->getRootNode();
        if(rank == 0)
        {
            gvt::core::DBNodeH dataNodes = cntxt->addToSync(cntxt-createNodeFromType("Data","Data",root.UUID()));
            // this node holds instances if any
            cntxt->addToSync(cntxt->createNodeFromType("Instances","Instances",root.UUID()));
        }
        cntxt->syncContext();
        // default camera and light nodes

    }   
}
void addMesh(int loc, Box3D *mshbx, Mesh *mesh,String meshname) {
    // add a mesh to the context.
    // loc is the rank the mesh lives on
    // mshbx is the mesh bounding box
    // mesh is the mesh itself
    
    // grab the context
    gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
    // get the data node.
    gvt::core::DBNodeH root = cntxt->getRootNode();
    gvt::core::DBNodeH dataNodes = root["Data"];
    // meshes get appended to the data node as a child
    gvt::core::DBNodeH amesh=cntxt->createNodeFromType("Mesh",meshname,dataNodes.UUID());
    amesh["file"] = meshname;
    amesh["bbox"] = (unsigned long long)meshbox;
    amesh["ptr"] = (unsigned long long)mesh;
    int rank ;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    gvt::core::DBNodeH loc = cntxt->createNode("rank",rank);
    amesh["Locations"] += loc;
    cntxt->addToSync(amesh);
}
void addInstanceToContext() {} // pnav: context has only single instance... what is this supposed to do?
/* add a point light to the render context 
 * \param name the name of the light
 * \param pos the light location in world coordinates
 * \param color the light color as RGB float
 */
void addLight(String name, glm::vec3& pos, glm::vec3& color) {
    gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
    gvt::core::DBNodeH root = ctx->getRootNode();
    gvt::core::DBNodeH lights = root["Lights"];
    gvt::core::DBNodeH pl = ctx->createNodeFromType("PointLight",name,lights.UUID());
    pl["position"] = pos;
    pl["color"] = color;
    
    ctx->addToSync(pl);
}
/* add an area light to the render context 
 * \param name the name of the light
 * \param pos the light location in world coordinates
 * \param color the light color as RGB float
 * \param n the normal for the area light surface
 * \param w the area light width
 * \param h the area light height
 */
void addLight(String name, glm::vec3& pos, glm::vec3& color, glm::vec3& n, float w, float h) {
    gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
    gvt::core::DBNodeH root = ctx->getRootNode();
    gvt::core::DBNodeH lights = root["Lights"];
    gvt::core::DBNodeH al = ctx->createNodeFromType("AreaLight",name,lights.UUID());
    al["position"] = pos;
    al["color"] = color;
    al["normal"] = n;
    al["height"] = h;
    al["width"] = w;

    ctx->addToSync(al);
}
/* modify an existing light position and/or color. This works for PointLight and AreaLight objects. If the light does not exist, this method has no effect. An error message will be printed if compiled with debugging.
 * \param name the name of the light
 * \param pos the new light positon
 * \param color the new light color
 */
void modifyLight(String name, glm::vec3& pos, glm::vec3& color) {
    gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
    gvt::core::DBNodeH root = ctx->getRootNode();
    gvt::core::DBNodeH lights = root.getChildByName("Lights");  // use this search so new node is not created if no lights
    if (!lights.isValid()) { GVT_ERR_MESSAGE("modifyLight() called for light " << name << " but no lights defined"); return; }
    gvt::core::DBNodeH light = root.getChildByName(name);
    if (!light.isValid()) { GVT_ERR_MESSAGE("modifyLight() called for light " << name << " but no such light found"); return; }

    light["position"] = pos;
    light["color"] = color;

    ctx->addToSync(light);     
}
/* modify an existing light position, color, normal, height and/or width. Calling this on a PointLight will make it an AreaLight. If the light does not exist, this method has no effect. An error message will be printed if compiled with debugging.
 * \param name the name of the light
 * \param pos the new light positon
 * \param color the new light color
 * \param n the new normal
 * \param w the new width
 * \param h the new height 
 */
void modifyLight(String name, glm::vec3& pos, glm::vec3& color, glm::vec3& n, float w, float h) {
    gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
    gvt::core::DBNodeH root = ctx->getRootNode();
    gvt::core::DBNodeH lights = root.getChildByName("Lights");  // use this search so new node is not created if no lights
    if (!lights.isValid()) { GVT_ERR_MESSAGE("modifyLight() called for light " << name << " but no lights defined"    ); return; }
    gvt::core::DBNodeH light = root.getChildByName(name);
    if (!light.isValid()) { GVT_ERR_MESSAGE("modifyLight() called for light " << name << " but no such light found    "); return; }

    light["position"] = pos;
    light["color"] = color;
    light["normal"] = n;
    light["height"] = h;
    light["width"] = w;

    ctx->addToSync(light);
}     
/* add a camera to the scene
 * \param name the camera name
 * \param pos the camera position
 * \param focus the focus direction of the camera
 * \param up the up orientation vector for the camera
 * \param fov the camera field of view in radians
 * \param depth the maximum ray depth for rays spawned from this camera
 * \param samples the number of rays cast per pixel for this camera
 * \param jitter the window size for jittering multiple samples per pixel
 */
void addCamera(String name, glm::vec3& pos, glm::vec3& focus, glm::vec3& up, float fov, int depth, int samples, float jitter) {
  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH root = ctx->getRootNode();
  gvt::core::DBNodeH cam = ctx->createNodeFromType("Camera", name, root.UUID());
  cam["eyePoint"] = pos;
  cam["focus"] = focus;
  cam["upVector"] = up;
  cam["fov"] = fov;
  cam["rayMaxDepth"] = depth;
  cam["raySamples"] = samples;
  cam["jitterWindowSize"] = jitter; 

  ctx->addToSync(cam);
}
/* modify the given camera, if it exists
 * \param name the camera name
 * \param pos the camera position
 * \param focus the focus direction of the camera
 * \param up the up orientation vector for the camera
 * \param fov the camera field of view in radians
 * \param depth the maximum ray depth for rays spawned from this camera
 * \param samples the number of rays cast per pixel for this camera
 * \param jitter the window size for jittering multiple samples per pixel
 */
void modifyCamera(String name, glm::vec3& pos, glm::vec3& focus, glm::vec3& up, float fov, int depth, int samples, float jitter) {
  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH root = ctx->getRootNode();
  gvt::core::DBNodeH cam = root.getChildByName(name);
  if (!cam.isValid()) { GVT_ERR_MESSAGE("modifyCamera() called for camera " << name << " but no camera defined"); return; }
  cam["eyePoint"] = pos;
  cam["focus"] = focus; 
  cam["upVector"] = up;
  cam["fov"] = fov;
  cam["rayMaxDepth"] = depth;
  cam["raySamples"] = samples;
  cam["jitterWindowSize"] = jitter; 

  ctx->addToSync(cam);
}
/* modify the given camera, if it exists
 * \param name the camera name
 * \param pos the camera position
 * \param focus the focus direction of the camera
 * \param up the up orientation vector for the camera
 * \param fov the camera field of view in radians
 */
void modifyCamera(String name, glm::vec3& pos, glm::vec3& focus, glm::vec3& up, float fov) {
  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH root = ctx->getRootNode();
  gvt::core::DBNodeH cam = root.getChildByName(name);
  if (!cam.isValid()) { GVT_ERR_MESSAGE("modifyCamera() called for camera " << name << " but no such camera defined"); return; }
  cam["eyePoint"] = pos;
  cam["focus"] = focus; 
  cam["upVector"] = up;
  cam["fov"] = fov;

  ctx->addToSync(cam);
}
/* add a film object to the context
 * \param w the image width
 * \param h the image height
 * \param path the path for the image file
 */
void addFilm(String name, int w, int h, String path) {
  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH root = ctx->getRootNode();
  gvt::core::DBNodeH film = ctx->createNodeFromType("Film", name, root.UUID());
  film["width"] = w;
  film["height"] = h;
  film["outputPath"] = path;

  ctx->addToSync(film);
}
/* modify film object, if it exists
 * \param w the image width
 * \param h the image height
 * \param path the path for the image file
 */
void modifyFilm(String name, int w, int h, String path) {
  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH root = ctx->getRootNode();
  gvt::core::DBNodeH film = root.getChildByName(name);
  if (!film.isValid()) { GVT_ERR_MESSAGE("modifyFilm() called for film " << name << " but no such film defined"); return; }
  film["width"] = w;
  film["height"] = h;
  film["outputPath"] = path;

  ctx->addToSync(film);
}
/* add a renderer to the context
 * \param name the renderer name
 * \param adapter the rendering adapter / engine used (ospray,embree,optix,manta)
 * \param schedule the schedule to use for this adapter (image,domain,hybrid)
 */
void addRenderer(String name, int adapter, int schedule) {
  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH root = ctx->getRootNode();
  gvt::core::DBNodeH render = ctx->createNodeFromType("Schedule", name, root.UUID());
  render["type"] = schedule;
  render["adapter"] = adapter;

  ctx->addToSync(render);
}
/* modify a renderer in the context, if it exists
 * \param name the renderer name
 * \param adapter the rendering adapter / engine used (ospray,embree,optix,manta)
 * \param schedule the schedule to use for this adapter (image,domain,hybrid)
 */
void modifyRenderer(String name, int adapter, int schedule) {
  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH root = ctx->getRootNode();
  gvt::core::DBNodeH render = root.getChildByName(name);
  if (!render.isValid()) { GVT_ERR_MESSAGE("modifyRenderer() called for renderer " << name << " but no such renderer defined"); return; }
  render["type"] = schedule;
  render["adapter"] = adapter;

  ctx->addToSync(render);
}
