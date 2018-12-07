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

#include <string>
#include <thread>

namespace api {

void gvtInit(int argc, char **argv, unsigned int threads = std::thread::hardware_concurrency());
// void addMesh(gvt::render::data::primitives::Box3D *mshbx, gvt::render::data::primitives::Mesh *mesh,
//              std::string meshname);

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

/**
 * Add color to vertex
 *
 * @param name Name of the mesh
 * @param n Number of vertex
 * @param kd Color 3 * float per vertex (rgb)
 */
void addMeshVertexColor(const std::string name, const unsigned n, const float *kd);

/**
 * each mesh needs one or more instance.
 * Insert an instance for a particular named mesh. The instance
 * contains the transformation matrix information needed to locate
 * the mesh correctly in world space
 * \param meshname the name of the mesh this instance refers to.
 * \param instId id of this instance
 * \param m transformation matrix that moves and scales instance*/

void addInstance(std::string instname, std::string meshname, const float *m);

#ifdef GVT_BUILD_VOLUME
/**
 * Creates a volume with a unique name
 * volumes use the same nodes as a mesh
 * Can contain amr volume
 * \param name unique name of the mesh
 * \param amr automatic mesh refinement flag
 */
void createVolume(const std::string name,const bool amr);
/**
 * Add transfer function to the volume
 * \param name : volume unique identifier
 * \param colortfname : string name of color transfer function file
 * \param opacitytfname : string name of opacity transfer function file
 * \param low : lower scalar value 
 * \param high : upper scalar value
 * */
void addVolumeTransferFunctions(const std::string name, const std::string colortfname, const std::string opacitytfname,float low,float high);
/**
 * this function adds sample data and necessary data to the volume object
 * \param name the name of the volume node
 * \param samples the pointer to the sample data
 * \param counts the dimensions of the sample data
 * \param deltas the spacing of the sample data
 * \param samplingrate the number of samples per cell used to integrate
 */
void addVolumeSamples(const std::string name,  float *samples,  int *counts,  float *origin,  float *deltas,  float samplingrate);
/**
 * this function adds a subgrid to the existing volume object
 * \param name the name of the volume node
 * \param samples the pointer to the sample data
 * \param counts the dimensions of the sample data
 * \param deltas the spacing of the sample data
 * \param samplingrate the number of samples per cell used to integrate
 */
void addAmrSubgrid(const std::string name,int gridid, float *samples, int *counts, float *origin, float *deltas);
#endif // GVT_BUILD_VOLUME

/**
 * add a point light to the render context
 * \param name the name of the light
 * \param pos the light location in world coordinates
 * \param color the light color as RGB float
 */
void addPointLight(std::string name, const float *pos, const float *color);

/**
 * add an area light to the render context
 * \param name the name of the light
 * \param pos the light location in world coordinates
 * \param color the light color as RGB float
 * \param n the normal for the area light surface
 * \param w the area light width
 * \param h the area light height
 */
void addAreaLight(std::string name, const float *pos, const float *color, const float *n, float w, float h);

/**
 * modify an existing light position and/or color. This works for PointLight and AreaLight objects. If the light does
 * not exist, this method has no effect. An error message will be printed if compiled with debugging. \param name the
 * name of the light \param pos the new light positon \param color the new light color
 */
void modifyLight(std::string name, const float *pos, const float *color);

/**
 * modify an existing light position, color, normal, height and/or width. Calling this on a PointLight will make it an
 * AreaLight. If the light does not exist, this method has no effect. An error message will be printed if compiled with
 * debugging. \param name the name of the light \param pos the new light positon \param color the new light color \param
 * n the new normal \param w the new width \param h the new height
 */
void modifyLight(std::string name, const float *pos, const float *color, const float *n, float w, float h);

/**
 * add a camera to the scene
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

/**
 * modify the given camera, if it exists
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

/**
 * modify the given camera, if it exists
 * \param name the camera name
 * \param pos the camera position
 * \param focus the focus direction of the camera
 * \param up the up orientation vector for the camera
 * \param fov the camera field of view in radians
 */
void modifyCamera(std::string name, const float *pos, const float *focus, const float *up, float fov);

/**
 * add a film object to the context
 * \param w the image width
 * \param h the image height
 * \param path the path for the image file
 */
void addFilm(std::string name, int w, int h, std::string path);

/**
 * modify film object, if it exists
 * \param w the image width
 * \param h the image height
 * \param path the path for the image file
 */
void modifyFilm(std::string name, int w, int h, std::string path);

/**
 * add a renderer to the context
 * \param name the renderer name
 * \param adapter the rendering adapter / engine used (ospray,embree,optix,manta)
 * \param schedule the schedule to use for this adapter (image,domain,hybrid)
 */
void render(std::string name);

/**
 * synchronize state changes to remote processes
 */
void gvtsync();

void writeimage(std::string name, std::string output = "");

void addRenderer(std::string name, int adapter, int schedule,  std::string const& Camera = "Camera", std::string const& Film = "Film", bool volume = false);

/**
 * modify a renderer in the context, if it exists
 * \param name the renderer name
 * \param adapter the rendering adapter / engine used (ospray,embree,optix,manta)
 * \param schedule the schedule to use for this adapter (image,domain,hybrid)
 */
void modifyRenderer(std::string name, int adapter, int schedule, std::string const& Camera = "Camera", std::string const& Film = "Film");
} // namespace api
#endif // GVT_RENDER_API_H
