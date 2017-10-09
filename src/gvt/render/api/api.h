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

#ifndef GVT_RENDER_API_H
#define GVT_RENDER_API_H
// API functions
// #include <gvt/core/Math.h>
// #include <gvt/core/context/Variant.h>
// #include <gvt/render/RenderContext.h>
// #include <gvt/render/Schedulers.h>
// #include <gvt/render/Types.h>
// #include <gvt/render/data/Domains.h>
#include <string>

// #include <tbb/task_scheduler_init.h>
// #include <thread>
//
// #ifdef GVT_RENDER_ADAPTER_EMBREE
// #include <gvt/render/adapter/embree/EmbreeMeshAdapter.h>
// #endif
//
// #ifdef GVT_RENDER_ADAPTER_MANTA
// #include <gvt/render/adapter/manta/MantaMeshAdapter.h>
// #endif
//
// #ifdef GVT_RENDER_ADAPTER_OPTIX
// #include <gvt/render/adapter/optix/OptixMeshAdapter.h>
// #endif

// using namespace std;
// using namespace gvt::render::data::primitives;

void gvtInit(int argc, char **argv);
// void addMesh(gvt::render::data::primitives::Box3D *mshbx, gvt::render::data::primitives::Mesh *mesh,
//              std::string meshname);

/* Reads ply files in a directory.
 * \param[in] dirname Name of directory containing ply files.
 * \param[in] dist True if multiple nodes are used.
 * \param[out] world_bounds Min/Max coordinates for world bounds. bounds_min(x,y,z) and bounds_max(x,y,z).
 */
void readPly(const std::string dirname, bool dist, float *world_bounds);


/* Creates a mesh with a unique name
 * \param Mesh unique name
 */
void createMesh(const std::string name);

/* Add vertices array to the mesh
 * \param name : mesh unique identifier
 * \param n : number of vertices
 * \param vertices : mesh vertices consecutive <x,y,x>
 */
void addMeshVertices(const std::string name, const unsigned &n, const float *vertices);

/* Add triangles array to the mesh
 * \param name : mesh unique identifier
 * \param n : number of triangles
 * \param triangle : ccw triangle vertex indices <a,b,c>
 */
void addMeshTriangles(const std::string name, const unsigned &n, const unsigned *triangles);

/* Add triangles face normals array to the mesh
 * \param name : mesh unique identifier
 * \param n : number of triangles
 * \param normals : face normals <x,y,z>
 */
void addMeshFaceNormals(const std::string name, const unsigned &n, const float *normals);

/* Add triangles face normals array to the mesh
 * \param name : mesh unique identifier
 * \param n : number of vertex
 * \param normals : vertex normals <x,y,z>
 */
void addMeshVertexNormals(const std::string name, const unsigned &n, const float *normals);

/* Finish mesh by computing bounds and normals
 * \param compute_normals : Compute normals at each vertex
 */
void finishMesh(const std::string name, const bool compute_normal = true);

/**
 * Add global diffuse material to mesh
 *
 */
void addMeshMaterial(const std::string name, const unsigned mattype, const float *kd, const float alpha = 1.f);

/**
 * Add global specular material to mesh
 *
 */
void addMeshMaterial(const std::string name, const unsigned mattype, const float *kd, const float *ks,
                     const float alpha = 1.f);

/**
 * Add material list to mesh
 *
 *
 */
void addMeshMaterials(const std::string name, const unsigned n, const unsigned *mattype, const float *kd,
                      const float *ks, const float *alpha);

/* each mesh needs one or more instance.
 * Insert an instance for a particular named mesh. The instance
 * contains the transformation matrix information needed to locate
 * the mesh correctly in world space
 * \param meshname the name of the mesh this instance refers to.
 * \param instId id of this instance
 * \param m transformation matrix that moves and scales instance*/

void addInstance(std::string name, const float *m);

/* add a point light to the render context
 * \param name the name of the light
 * \param pos the light location in world coordinates
 * \param color the light color as RGB float
 */
void addPointLight(std::string name, const float *pos, const float *color);

/* add an area light to the render context
 * \param name the name of the light
 * \param pos the light location in world coordinates
 * \param color the light color as RGB float
 * \param n the normal for the area light surface
 * \param w the area light width
 * \param h the area light height
 */
void addAreaLight(std::string name, const float *pos, const float *color, const float *n, float w, float h);
/* modify an existing light position and/or color. This works for PointLight and AreaLight objects. If the light does
 * not exist, this method has no effect. An error message will be printed if compiled with debugging. \param name the
 * name of the light \param pos the new light positon \param color the new light color
 */
void modifyLight(std::string name, const float *pos, const float *color);
/* modify an existing light position, color, normal, height and/or width. Calling this on a PointLight will make it an
 * AreaLight. If the light does not exist, this method has no effect. An error message will be printed if compiled with
 * debugging. \param name the name of the light \param pos the new light positon \param color the new light color \param
 * n the new normal \param w the new width \param h the new height
 */
void modifyLight(std::string name, const float *pos, const float *color, const float *n, float w, float h);
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
void addCamera(std::string name, const float *pos, const float *focus, const float *up, float fov, int depth,
               int samples, float jitter);

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
void modifyCamera(std::string name, const float *pos, const float *focus, const float *up, float fov, int depth,
                  int samples, float jitter);
/* modify the given camera, if it exists
 * \param name the camera name
 * \param pos the camera position
 * \param focus the focus direction of the camera
 * \param up the up orientation vector for the camera
 * \param fov the camera field of view in radians
 */
void modifyCamera(std::string name, const float *pos, const float *focus, const float *up, float fov);
/* add a film object to the context
 * \param w the image width
 * \param h the image height
 * \param path the path for the image file
 */
void addFilm(std::string name, int w, int h, std::string path);
/* modify film object, if it exists
 * \param w the image width
 * \param h the image height
 * \param path the path for the image file
 */
void modifyFilm(std::string name, int w, int h, std::string path);
/* add a renderer to the context
 * \param name the renderer name
 * \param adapter the rendering adapter / engine used (ospray,embree,optix,manta)
 * \param schedule the schedule to use for this adapter (image,domain,hybrid)
 */

void render(std::string name);
void writeimage(std::string name, std::string output = "");

void addRenderer(std::string name, int adapter, int schedule);
/* modify a renderer in the context, if it exists
 * \param name the renderer name
 * \param adapter the rendering adapter / engine used (ospray,embree,optix,manta)
 * \param schedule the schedule to use for this adapter (image,domain,hybrid)
 */
void modifyRenderer(std::string name, int adapter, int schedule);
#endif // GVT_RENDER_API_H
