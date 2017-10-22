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
#include <assert.h>
#include <gvt/core/Math.h>
#include <gvt/core/context/Variant.h>
#include <gvt/render/RenderContext.h>
#include <gvt/render/Renderer.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/Types.h>
#include <gvt/render/data/Domains.h>

#include <string>

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

#include "api.h"
#include <gvt/render/cntx/rcontext.h>

using namespace std;
using namespace gvt::render::data::primitives;

namespace api2 {

void gvtInit(int argc, char **argv) {
  // init mpi... or not
  int initialized, rank;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }
  // initialize the context and a few other things.
  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  if (cntxt == NULL) // context creation failed
  {
    GVT_ERR_MESSAGE("gvtInit: context init failed");
    exit(0);
  }

  cntx::node &root = cntx::rcontext::instance().root();

  //  if(cntx::rcontext::instance().cntx_comm.rank == 0) {
  cntx::rcontext::instance().createnode_allranks("threads", "threads", true, root.getid());
  cntx::rcontext::instance().getUnique("threads") = (int)std::thread::hardware_concurrency() / 2;
  //  }

  //  cntx::rcontext::sync();

  // root node
  //  root += cntxt->createNode("threads", (int)std::thread::hardware_concurrency());
  //  // nodes that have multpple leaves are added here
  //  // for example there may be many lights but generally one camera.
  //  if (rank == 0) {
  //    // root node for mesh data
  //    cntxt->addToSync(cntxt->createNodeFromType("Data", "Data", root.UUID()));
  //    // this node holds instances if any
  //    cntxt->addToSync(cntxt->createNodeFromType("Instances", "Instances", root.UUID()));
  //    // root node for lights
  //    cntxt->addToSync(cntxt->createNodeFromType("Lights", "Lights", root.UUID()));
  //    // root node for camera
  //    // cntxt->addToSync(cntxt->createNodeFromType("Camera","Camera",root.UUID()));
  //    // root node for film
  //    // cntxt->addToSync(cntxt->createNodeFromType("Film","Film",root.UUID()));
  //    // root node for renderer
  //    // cntxt->addToSync(cntxt->createNodeFromType("Schedule","Schedule",root.UUID()));
  //  }
  //  cntxt->syncContext();
}

void createMesh(const std::string name) {

  cntx::rcontext &db = cntx::rcontext::instance();
  cntx::node &root = cntx::rcontext::instance().root();

  db.createnode("Mesh", name, true, db.getUnique("Data").getid());
  db.getChild(db.getUnique(name), "file") = name;
  db.getChild(db.getUnique(name), "ptr") = std::make_shared<gvt::render::data::primitives::Mesh>();
  db.getChild(db.getUnique(name), "bbox") = std::make_shared<gvt::render::data::primitives::Box3D>();
}

cntx::node &getChildByName(const cntx::node &node, std::string name) {
  cntx::rcontext &db = cntx::rcontext::instance();
  return db.getChild(node, name);
}

void printChildName(const cntx::node &node) {
  cntx::rcontext &db = cntx::rcontext::instance();
  for (auto &c : db.getChildren(node)) {
    std::cout << c.get().name << std::endl;
  }
}

void addMeshVertices(const std::string name, const unsigned &n, const float *vertices) {

  cntx::rcontext &db = cntx::rcontext::instance();
  std::shared_ptr<gvt::render::data::primitives::Mesh> m = getChildByName(db.getUnique(name), "ptr");

  for (int i = 0; i < n * 3; i += 3) {
    m->addVertex(glm::vec3(vertices[i + 0], vertices[i + 1], vertices[i + 2]));
  }
}

void addMeshTriangles(const std::string name, const unsigned &n, const unsigned int *triangles) {

  cntx::rcontext &db = cntx::rcontext::instance();
  std::shared_ptr<gvt::render::data::primitives::Mesh> m = getChildByName(db.getUnique(name), "ptr");
  for (int i = 0; i < n * 3; i += 3) {
    m->addFace(triangles[i + 0], triangles[i + 1], triangles[i + 2]);
  }
}

void finishMesh(const std::string name, const bool compute_normal) {
  cntx::rcontext &db = cntx::rcontext::instance();
  std::shared_ptr<gvt::render::data::primitives::Mesh> m = getChildByName(db.getUnique(name), "ptr");
  m->computeBoundingBox();
  if (compute_normal) m->generateNormals();
  cntx::node &bbnode = getChildByName(db.getUnique(name), "bbox");
  bbnode = std::make_shared<gvt::render::data::primitives::Box3D>(*m->getBoundingBox());
  std::shared_ptr<std::vector<int> > v = std::make_shared<std::vector<int> >();
  v->push_back(db.cntx_comm.rank);
  getChildByName(db.getUnique(name), "Locations") = v; // db.cntx_comm.rank;
}

/* Add triangles face normals array to the mesh
 * \param name : mesh unique identifier
 * \param n : number of triangles
 * \param normals : face normals <x,y,z>
 */
void addMeshFaceNormals(const std::string name, const unsigned &n, const float *normals) {
  cntx::rcontext &db = cntx::rcontext::instance();
  std::shared_ptr<gvt::render::data::primitives::Mesh> m = getChildByName(db.getUnique(name), "ptr");

  for (int i = 0; i < n * 3; i += 3) {
    m->face_normals.push_back(glm::vec3(normals[i + 0], normals[i + 1], normals[i + 2]));
  }
}

/* Add triangles face normals array to the mesh
 * \param name : mesh unique identifier
 * \param n : number of vertex
 * \param normals : vertex normals <x,y,z>
 */
void addMeshVertexNormals(const std::string name, const unsigned &n, const float *normals) {
  cntx::rcontext &db = cntx::rcontext::instance();
  std::shared_ptr<gvt::render::data::primitives::Mesh> m = getChildByName(db.getUnique(name), "ptr");

  for (int i = 0; i < n * 3; i += 3) {
    m->normals.push_back(glm::vec3(normals[i + 0], normals[i + 1], normals[i + 2]));
  }
}

/**
 * Add global diffuse material to mesh
 *
 */
void addMeshMaterial(const std::string name, const unsigned mattype, const float *kd, const float alpha) {
  cntx::rcontext &db = cntx::rcontext::instance();
  std::shared_ptr<gvt::render::data::primitives::Mesh> m = getChildByName(db.getUnique(name), "ptr");
  m->mat = new gvt::render::data::primitives::Material();
  m->mat->type = mattype;
  m->mat->kd = glm::vec3(kd[0], kd[1], kd[2]);
  m->mat->alpha = alpha;
}

/**
 * Add global specular material to mesh
 *
 */
void addMeshMaterial(const std::string name, const unsigned mattype, const float *kd, const float *ks,
                     const float alpha) {
  cntx::rcontext &db = cntx::rcontext::instance();
  std::shared_ptr<gvt::render::data::primitives::Mesh> m = getChildByName(db.getUnique(name), "ptr");
  m->mat = new gvt::render::data::primitives::Material();
  m->mat->type = mattype;
  m->mat->kd = glm::vec3(kd[0], kd[1], kd[2]);
  m->mat->ks = glm::vec3(ks[0], ks[1], ks[2]);
  m->mat->alpha = alpha;
}

/**
 * Add material list to mesh
 *
 *
 */
void addMeshMaterials(const std::string name, const unsigned n, const unsigned *mattype, const float *kd,
                      const float *ks, const float *alpha) {
  cntx::rcontext &db = cntx::rcontext::instance();
  std::shared_ptr<gvt::render::data::primitives::Mesh> m = getChildByName(db.getUnique(name), "ptr");

  for (int i = 0; i < n; i++) {
    gvt::render::data::primitives::Material *mat = new gvt::render::data::primitives::Material();
    mat->type = mattype[i];
    mat->kd = glm::vec3(kd[(i * 3) + 0], kd[(i * 3) + 1], kd[(i * 3) + 2]);
    mat->ks = glm::vec3(ks[(i * 3) + 0], ks[(i * 3) + 1], ks[(i * 3) + 2]);
    mat->alpha = alpha[i];
    m->faces_to_materials.push_back(mat);
  }
}

// void addMeshMaterial(const std::string name, const unsigned mattype, const float *kd, const float *ks) {
//   gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
//   // get the data node.
//   gvt::core::DBNodeH root = cntxt->getRootNode();
//   gvt::core::DBNodeH dataNodes = root["Data"];
//   GVT_ASSERT(dataNodes.getChildByName(name.c_str()).isValid(), "Mesh name is not unique : " << name);
//   gvt::core::DBNodeH ameshnode = dataNodes.getChildByName(name.c_str());
//
//   gvt::render::data::primitives::Mesh *m =
//       reinterpret_cast<gvt::render::data::primitives::Mesh *>(ameshnode["ptr"].value().toULongLong());
//
//   gvt::render::data::primitives::Material *mat = new gvt::render::data::primitives::Material();
//
//   mat->type = mattype;
//   mat->kd = glm::vec3(kd[0], kd[1], kd[2]);
//   mat->ks = glm::vec3(ks[0], ks[1], ks[2]);
//
//   m->setMaterial(mat);
// }
// void addMeshMaterials(const std::string name, const unsigned n, const unsigned *mattype, const float *kd,
//                       const float *ks) {}

// void addMesh(Box3D *mshbx, Mesh *mesh, string meshname) {
//  // add a mesh to the context.
//  // loc is the rank the mesh lives on
//  // mshbx is the mesh bounding box
//  // mesh is the mesh itself
//
//  // grab the context
//  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
//  // get the data node.
//  gvt::core::DBNodeH root = cntxt->getRootNode();
//  gvt::core::DBNodeH dataNodes = root["Data"];
//  // meshes get appended to the data node as a child
//  gvt::core::DBNodeH ameshnode = cntxt->createNodeFromType("Mesh", meshname.c_str(), dataNodes.UUID());
//  ameshnode["file"] = meshname.c_str();
//  ameshnode["bbox"] = (unsigned long long)mshbx;
//  ameshnode["ptr"] = (unsigned long long)mesh;
//  int rank;
//  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//  gvt::core::DBNodeH loc = cntxt->createNode("rank", rank);
//  ameshnode["Locations"] += loc;
//  cntxt->addToSync(ameshnode);
//}

/* each mesh needs one or more instance.
 * Insert an instance for a particular named mesh. The instance
 * contains the transformation matrix information needed to locate
 * the mesh correctly in world space
 * \param meshname the name of the mesh this instance refers to.
 * \param instId id of this instance
 * \param m transformation matrix that moves and scales instance*/

void addInstance(std::string instancename, std::string meshname, const float *am) {

  cntx::rcontext &db = cntx::rcontext::instance();

  cntx::node &ameshnode = db.getUnique(meshname);

  cntx::node &inode = db.createnode("Instance", instancename, true, db.getUnique("Instances").getid());

  // db.printtree(std::cout);

  std::shared_ptr<gvt::render::data::primitives::Box3D> mbox = getChildByName(ameshnode, "bbox");

  // build the instance data
  std::shared_ptr<glm::mat4> m = std::make_shared<glm::mat4>(1.f);
  *m = glm::make_mat4(am);
  std::shared_ptr<glm::mat4> minv = std::make_shared<glm::mat4>(1.f);
  std::shared_ptr<glm::mat3> normi = std::make_shared<glm::mat3>(1.f);
  *minv = glm::inverse(*m);
  *normi = glm::transpose(glm::inverse(glm::mat3(*m)));
  auto il = glm::vec3((*m) * glm::vec4(mbox->bounds_min, 1.f));
  auto ih = glm::vec3((*m) * glm::vec4(mbox->bounds_max, 1.f));
  std::shared_ptr<gvt::render::data::primitives::Box3D> ibox =
      std::make_shared<gvt::render::data::primitives::Box3D>(il, ih);

  getChildByName(inode, "id") = inode.name;
  getChildByName(inode, "meshRef") = ameshnode.getid();
  getChildByName(inode, "mat") = m;
  getChildByName(inode, "matinv") = minv;
  getChildByName(inode, "normi") = normi;
  getChildByName(inode, "bbox") = ibox;
  getChildByName(inode, "centroid") = ibox->centroid();
  //
  //  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  //  gvt::core::DBNodeH root = ctx->getRootNode();
  //  gvt::core::DBNodeH instNodes = root["Instances"];
  //  gvt::core::DBNodeH dataNodes = root["Data"];
  //  gvt::core::DBNodeH ameshnode = getChildByName(dataNodes, name);
  //  GVT_ASSERT(ameshnode.isValid(), "Mesh name does not exist : " << name);
  //
  //  int instID = instNodes.getChildren().size();
  //  std::string instname = name + std::to_string(instNodes.getChildren().size());
  //
  //  gvt::core::DBNodeH instNode = ctx->createNodeFromType("Instance", instname.c_str(), instNodes.UUID());
  //
  //  std::cout << "Create instance " << instname << std::endl;
  //  Box3D *mbox = (Box3D *)ameshnode["bbox"].value().toULongLong();
  //  glm::mat4 *m = new glm::mat4(1.f);
  //  *m = glm::make_mat4(am);
  //
  //  // build the instance data
  //  auto minv = new glm::mat4(1.f);
  //  auto normi = new glm::mat3(1.f);
  //  *minv = glm::inverse(*m);
  //  *normi = glm::transpose(glm::inverse(glm::mat3(*m)));
  //  auto il = glm::vec3((*m) * glm::vec4(mbox->bounds_min, 1.f));
  //  auto ih = glm::vec3((*m) * glm::vec4(mbox->bounds_max, 1.f));
  //  Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
  //  // load the node
  //  instNode["id"] = instID;
  //  instNode["meshRef"] = ameshnode.UUID();
  //  instNode["mat"] = (unsigned long long)m;
  //  instNode["matInv"] = (unsigned long long)minv;
  //  instNode["normi"] = (unsigned long long)normi;
  //  instNode["bbox"] = (unsigned long long)ibox;
  //  instNode["centroid"] = ibox->centroid();
  //
  //  ctx->addToSync(instNode);
}

/* add a point light to the render context
 * \param name the name of the light
 * \param pos the light location in world coordinates
 * \param color the light color as RGB float
 */
void addPointLight(string name, const float *pos, const float *color) {
  //  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  //  gvt::core::DBNodeH root = ctx->getRootNode();
  //  gvt::core::DBNodeH lights = root["Lights"];
  //  gvt::core::DBNodeH pl = ctx->createNodeFromType("PointLight", name, lights.UUID());
  //  pl["position"] = glm::make_vec3(pos);
  //  pl["color"] = glm::make_vec3(color);
  //
  //  ctx->addToSync(pl);
}

/* add an area light to the render context
 * \param name the name of the light
 * \param pos the light location in world coordinates
 * \param color the light color as RGB float
 * \param n the normal for the area light surface
 * \param w the area light width
 * \param h the area light height
 */
void addAreaLight(string name, const float *pos, const float *color, const float *n, float w, float h) {
  //  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  //  gvt::core::DBNodeH root = ctx->getRootNode();
  //  gvt::core::DBNodeH lights = root["Lights"];
  //  gvt::core::DBNodeH al = ctx->createNodeFromType("AreaLight", name, lights.UUID());
  //  al["position"] = glm::make_vec3(pos);
  //  al["color"] = glm::make_vec3(color);
  //  al["normal"] = glm::make_vec3(n);
  //  al["height"] = h;
  //  al["width"] = w;
  //
  //  ctx->addToSync(al);
}

/* modify an existing light position and/or color. This works for PointLight and AreaLight objects. If the light does
 * not exist, this method has no effect. An error message will be printed if compiled with debugging. \param name the
 * name of the light \param pos the new light positon \param color the new light color
 */
void modifyLight(string name, const float *pos, const float *color) {
  //  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  //  gvt::core::DBNodeH root = ctx->getRootNode();
  //  gvt::core::DBNodeH lights = root.getChildByName("Lights"); // use this search so new node is not created if no
  //  lights if (!lights.isValid()) {
  //    GVT_ERR_MESSAGE("modifyLight() called for light " << name << " but no lights defined");
  //    return;
  //  }
  //  gvt::core::DBNodeH light = root.getChildByName(name);
  //  if (!light.isValid()) {
  //    GVT_ERR_MESSAGE("modifyLight() called for light " << name << " but no such light found");
  //    return;
  //  }
  //
  //  light["position"] = glm::make_vec3(pos);
  //  light["color"] = glm::make_vec3(color);
  //
  //  ctx->addToSync(light);
}

/* modify an existing light position, color, normal, height and/or width. Calling this on a PointLight will make it an
 * AreaLight. If the light does not exist, this method has no effect. An error message will be printed if compiled with
 * debugging. \param name the name of the light \param pos the new light positon \param color the new light color \param
 * n the new normal \param w the new width \param h the new height
 */
void modifyLight(string name, const float *pos, const float *color, const float *n, float w, float h) {
  //  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  //  gvt::core::DBNodeH root = ctx->getRootNode();
  //  gvt::core::DBNodeH lights = root.getChildByName("Lights"); // use this search so new node is not created if no
  //  lights if (!lights.isValid()) {
  //    GVT_ERR_MESSAGE("modifyLight() called for light " << name << " but no lights defined");
  //    return;
  //  }
  //  gvt::core::DBNodeH light = root.getChildByName(name);
  //  if (!light.isValid()) {
  //    GVT_ERR_MESSAGE("modifyLight() called for light " << name << " but no such light found    ");
  //    return;
  //  }
  //
  //  light["position"] = glm::make_vec3(pos);
  //  light["color"] = glm::make_vec3(color);
  //  light["normal"] = glm::make_vec3(n);
  //  light["height"] = h;
  //  light["width"] = w;
  //
  //  ctx->addToSync(light);
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
void addCamera(string name, const float *pos, const float *focus, const float *up, float fov, int depth, int samples,
               float jitter) {
  //  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  //  gvt::core::DBNodeH root = ctx->getRootNode();
  //  gvt::core::DBNodeH cam = ctx->createNodeFromType("Camera", name, root.UUID());
  //  cam["eyePoint"] = glm::make_vec3(pos);
  //  cam["focus"] = glm::make_vec3(focus);
  //  cam["upVector"] = glm::make_vec3(up);
  //  cam["fov"] = fov;
  //  cam["rayMaxDepth"] = depth;
  //  cam["raySamples"] = samples;
  //  cam["jitterWindowSize"] = jitter;
  //
  //  ctx->addToSync(cam);
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
void modifyCamera(string name, const float *pos, const float *focus, const float *up, float fov, int depth, int samples,
                  float jitter) {
  //  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  //  gvt::core::DBNodeH root = ctx->getRootNode();
  //  gvt::core::DBNodeH cam = root.getChildByName(name);
  //  if (!cam.isValid()) {
  //    GVT_ERR_MESSAGE("modifyCamera() called for camera " << name << " but no camera defined");
  //    return;
  //  }
  //  cam["eyePoint"] = glm::make_vec3(pos);
  //  cam["focus"] = glm::make_vec3(focus);
  //  cam["upVector"] = glm::make_vec3(up);
  //  cam["fov"] = fov;
  //  cam["rayMaxDepth"] = depth;
  //  cam["raySamples"] = samples;
  //  cam["jitterWindowSize"] = jitter;
  //
  //  ctx->addToSync(cam);
}

/* modify the given camera, if it exists
 * \param name the camera name
 * \param pos the camera position
 * \param focus the focus direction of the camera
 * \param up the up orientation vector for the camera
 * \param fov the camera field of view in radians
 */
void modifyCamera(string name, const float *pos, const float *focus, const float *up, float fov) {
  //  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  //  gvt::core::DBNodeH root = ctx->getRootNode();
  //  gvt::core::DBNodeH cam = root.getChildByName(name);
  //  if (!cam.isValid()) {
  //    GVT_ERR_MESSAGE("modifyCamera() called for camera " << name << " but no such camera defined");
  //    return;
  //  }
  //  cam["eyePoint"] = glm::make_vec3(pos);
  //  cam["focus"] = glm::make_vec3(focus);
  //  cam["upVector"] = glm::make_vec3(up);
  //  cam["fov"] = fov;
  //
  //  ctx->addToSync(cam);
}

/* add a film object to the context
 * \param w the image width
 * \param h the image height
 * \param path the path for the image file
 */
void addFilm(string name, int w, int h, string path) {
  //  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  //  gvt::core::DBNodeH root = ctx->getRootNode();
  //  gvt::core::DBNodeH film = ctx->createNodeFromType("Film", name, root.UUID());
  //  film["width"] = w;
  //  film["height"] = h;
  //  film["outputPath"] = path;
  //  ctx->addToSync(film);
}

/* modify film object, if it exists
 * \param w the image width
 * \param h the image height
 * \param path the path for the image file
 */
void modifyFilm(string name, int w, int h, string path) {
  //  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  //  gvt::core::DBNodeH root = ctx->getRootNode();
  //  gvt::core::DBNodeH film = root.getChildByName(name);
  //  if (!film.isValid()) {
  //    GVT_ERR_MESSAGE("modifyFilm() called for film " << name << " but no such film defined");
  //    return;
  //  }
  //  film["width"] = w;
  //  film["height"] = h;
  //  film["outputPath"] = path;
  //
  //  ctx->addToSync(film);
}

void render(std::string name) {
  //  gvt::render::gvtRenderer *ren = gvt::render::gvtRenderer::instance();
  //  ren->render();
}

void writeimage(std::string name, std::string output) {
  //  gvt::render::gvtRenderer *ren = gvt::render::gvtRenderer::instance();
  //  ren->WriteImage();
}

/* add a renderer to the context
 * \param name the renderer name
 * \param adapter the rendering adapter / engine used (ospray,embree,optix,manta)
 * \param schedule the schedule to use for this adapter (image,domain,hybrid)
 */
void addRenderer(string name, int adapter, int schedule) {
  //  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  //  gvt::core::DBNodeH root = ctx->getRootNode();
  //  gvt::core::DBNodeH render = ctx->createNodeFromType("Schedule", name, root.UUID());
  //  render["type"] = schedule;
  //  render["adapter"] = adapter;
  //
  //  ctx->addToSync(render);
}

/* modify a renderer in the context, if it exists
 * \param name the renderer name
 * \param adapter the rendering adapter / engine used (ospray,embree,optix,manta)
 * \param schedule the schedule to use for this adapter (image,domain,hybrid)
 */
void modifyRenderer(string name, int adapter, int schedule) {
  //  gvt::render::RenderContext *ctx = gvt::render::RenderContext::instance();
  //  gvt::core::DBNodeH root = ctx->getRootNode();
  //  gvt::core::DBNodeH render = root.getChildByName(name);
  //  if (!render.isValid()) {
  //    GVT_ERR_MESSAGE("modifyRenderer() called for renderer " << name << " but no such renderer defined");
  //    return;
  //  }
  //  render["type"] = schedule;
  //  render["adapter"] = adapter;
  //
  //  ctx->addToSync(render);
}
} // namespace api2