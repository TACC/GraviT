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

// Qhull bits
#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/RboxPoints.h"
#include "libqhullcpp/QhullError.h"
#include "libqhullcpp/QhullQh.h"
#include "libqhullcpp/QhullFacet.h"
#include "libqhullcpp/QhullFacetList.h"
#include "libqhullcpp/QhullLinkedList.h"
#include "libqhullcpp/QhullVertex.h"
#include "libqhullcpp/QhullPoint.h"
#include "libqhullcpp/QhullVertexSet.h"

// API functions
#include <cassert>
#include <gvt/core/Math.h>
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

#ifdef GVT_RENDER_ADAPTER_GALAXY
#include <gvt/render/adapter/galaxy/PVolAdapter.h>
#endif

#include "api.h"
#include <gvt/render/cntx/rcontext.h>

using namespace std;
using namespace gvt::render::data::primitives;
using orgQhull::Qhull;
using orgQhull::QhullFacetList;
using orgQhull::QhullFacet;
using orgQhull::QhullVertexSet;
using orgQhull::QhullVertexSetIterator;
using orgQhull::QhullVertex;
using orgQhull::QhullPoint;

namespace api {

void gvtInit(int argc, char **argv, unsigned int threads) {
  int initialized, rank;

  //gvt::comm::scomm::init(argc,argv);

  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  }

  cntx::node &root = cntx::rcontext::instance().root();
  cntx::rcontext::instance().createnode_allranks("threads","threads",true,root.getid());
  cntx::rcontext::instance().getUnique("threads") = threads;

#ifdef GVT_RENDER_ADAPTER_OSPRAY
  gvt::render::adapter::ospray::data::OSPRayAdapter::initospray(&argc, argv);
#endif

#ifdef GVT_RENDER_ADAPTER_GALAXY
  gvt::render::adapter::galaxy::data::PVolAdapter::init_pvol(&argc, argv);
#endif


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

void createMesh(const std::string name) {
  cntx::rcontext &db = cntx::rcontext::instance();
  cntx::node &root = cntx::rcontext::instance().root();
  db.createnode("Mesh", name, true, db.getUnique("Data").getid());
  db.getChild(db.getUnique(name), "file") = name;
  db.getChild(db.getUnique(name), "ptr") = std::make_shared<gvt::render::data::primitives::Mesh>();
  db.getChild(db.getUnique(name), "bbox") = std::make_shared<gvt::render::data::primitives::Box3D>();
}


void addMeshVertices(const std::string name, const unsigned &n, const float *vertices, const bool tesselate, const std::string qhullargs) {

  // it is assumed that we have 3D data.
  int dimension = 3;
  Qhull qhull;
  std::string control(qhullargs);
  cntx::rcontext &db = cntx::rcontext::instance();
  std::shared_ptr<gvt::render::data::primitives::Mesh> m = getChildByName(db.getUnique(name), "ptr");

  // qhull expects double  verticies and gvt uses float so make a temp array to
  // hold the doubles and delete it after the routines are done. What a waste.
  double *dverts = new double[3*n];
  for (int i = 0; i < n * 3; i += 3) {
    m->addVertex(glm::vec3(vertices[i], vertices[i + 1], vertices[i + 2]));
    //std::cerr << dverts[i] << " " << dverts[i+1] << " " << dverts[i+2] << std::endl;
  }
  if(tesselate) { // call qhull to tesselate the vertices and create the triangle mesh
      if(control.empty())
          control = "d Qz";
      for(int i=0;i<n*3;i+=3){
        dverts[i] = vertices[i];
        dverts[i+1] = vertices[i+1];
        dverts[i+2] = vertices[i+2];
      }
      // call qhull to tesselate
      qhull.runQhull("",dimension,n,dverts,control.c_str());
      delete dverts;
      // pull the tessellation data out of qhull and load it into gravit
      QhullFacetList facets = qhull.facetList();
      for(QhullFacetList::const_iterator i = facets.begin();i!=facets.end();++i){
        QhullFacet f = *i;
        if(facets.isSelectAll() || f.isGood()) {
          QhullVertexSet vs = f.vertices();
          QhullVertexSetIterator j = vs;
          if(!vs.isEmpty()) {
            QhullVertex v;
            QhullPoint p;
            if(vs.count() == 3) { // add a triangle
                m->addFace(vs[0].point().id()+1,vs[1].point().id()+1,
                        vs[2].point().id()+1);
            }
          }
        }
      }
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
  db.getChild(db.getUnique(name), "Locations") = v; // db.cntx_comm.rank;
}

/**
 * Add triangles face normals array to the mesh
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

/**
 * Add triangles face normals array to the mesh
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


void addMeshVertexColor(const std::string name, const unsigned n, const float *kd) {

  cntx::rcontext &db = cntx::rcontext::instance();
  std::shared_ptr<gvt::render::data::primitives::Mesh> m = getChildByName(db.getUnique(name), "ptr");

  for(int i=0; i <n ; i++) {
    m->addVertexColor(glm::vec3(kd[i*3+0],kd[i*3+1],kd[i*3+2]));
  }
}

/**
 * each mesh needs one or more instance.
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

}

/**
 * add a point light to the render context
 * \param name the name of the light
 * \param pos the light location in world coordinates
 * \param color the light color as RGB float
 */
void addPointLight(string name, const float *pos, const float *color) {
  cntx::rcontext &db = cntx::rcontext::instance();
  auto& l = db.createnode("PointLight",name,true,db.getUnique("Lights"));
  db.getChild(l,"position") = glm::make_vec3(pos);
  db.getChild(l,"color") = glm::make_vec3(color);
}

/**
 * add an area light to the render context
 * \param name the name of the light
 * \param pos the light location in world coordinates
 * \param color the light color as RGB float
 * \param n the normal for the area light surface
 * \param w the area light width
 * \param h the area light height
 */
void addAreaLight(string name, const float *pos, const float *color, const float *n, float w, float h) {
  cntx::rcontext &db = cntx::rcontext::instance();
  auto& l = db.createnode("AreaLight",name,true,db.getUnique("Lights"));
  db.getChild(l,"position") = glm::make_vec3(pos);
  db.getChild(l,"color") = glm::make_vec3(color);
  db.getChild(l,"normal") = glm::make_vec3(n);
  db.getChild(l,"height") = h;
  db.getChild(l,"width") = w;
}

/**
 * modify an existing light position and/or color. This works for PointLight and AreaLight objects. If the light does
 * not exist, this method has no effect. An error message will be printed if compiled with debugging. \param name the
 * name of the light \param pos the new light positon \param color the new light color
 */
void modifyLight(string name, const float *pos, const float *color) {
  cntx::rcontext &db = cntx::rcontext::instance();
  auto& l = db.getUnique(name);
  if(l.getid().isInvalid()) {
    return;
  }
  db.getChild(l,"position") = glm::make_vec3(pos);
  db.getChild(l,"color") = glm::make_vec3(color);
}

/**
 * modify an existing light position, color, normal, height and/or width. Calling this on a PointLight will make it an
 * AreaLight. If the light does not exist, this method has no effect. An error message will be printed if compiled with
 * debugging. \param name the name of the light \param pos the new light positon \param color the new light color \param
 * n the new normal \param w the new width \param h the new height
 */
void modifyLight(string name, const float *pos, const float *color, const float *n, float w, float h) {
  cntx::rcontext &db = cntx::rcontext::instance();
  auto& l = db.getUnique(name);
  if(l.getid().isInvalid()) {
    return;
  }
  db.getChild(l,"position") = glm::make_vec3(pos);
  db.getChild(l,"color") = glm::make_vec3(color);
  db.getChild(l,"normal") = glm::make_vec3(n);
  db.getChild(l,"height") = h;
  db.getChild(l,"width") = w;
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

  cntx::rcontext &db = cntx::rcontext::instance();
  auto& c = db.createnode("Camera",name,true,db.getUnique("Cameras"));
  db.getChild(c,"eyePoint") = glm::make_vec3(pos);
  db.getChild(c,"focus") = glm::make_vec3(focus);
  db.getChild(c,"upVector") = glm::make_vec3(up);
  db.getChild(c,"fov") = fov;
  db.getChild(c,"rayMaxDepth") = depth;
  db.getChild(c,"raySamples") = samples;
  db.getChild(c,"jitterWindowSize") =jitter;

}

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
void modifyCamera(string name, const float *pos, const float *focus, const float *up, float fov, int depth, int samples,
                  float jitter) {
  cntx::rcontext &db = cntx::rcontext::instance();
  auto& c = db.getUnique(name);
  if(c.getid().isInvalid()) {
    return;
  }
  db.getChild(c,"eyePoint") = glm::make_vec3(pos);
  db.getChild(c,"focus") = glm::make_vec3(focus);
  db.getChild(c,"upVector") = glm::make_vec3(up);
  db.getChild(c,"fov") = fov;
  db.getChild(c,"rayMaxDepth") = depth;
  db.getChild(c,"raySamples") = samples;
  db.getChild(c,"jitterWindowSize") =jitter;
}

/**
 * modify the given camera, if it exists
 * \param name the camera name
 * \param pos the camera position
 * \param focus the focus direction of the camera
 * \param up the up orientation vector for the camera
 * \param fov the camera field of view in radians
 */
void modifyCamera(string name, const float *pos, const float *focus, const float *up, float fov) {
  cntx::rcontext &db = cntx::rcontext::instance();
  auto& c = db.getUnique(name);
  if(c.getid().isInvalid()) {
    return;
  }
  db.getChild(c,"eyePoint") = glm::make_vec3(pos);
  db.getChild(c,"focus") = glm::make_vec3(focus);
  db.getChild(c,"upVector") = glm::make_vec3(up);
  db.getChild(c,"fov") = fov;
}

/**
 * add a film object to the context
 * \param w the image width
 * \param h the image height
 * \param path the path for the image file
 */
void addFilm(string name, int w, int h, string path) {
  cntx::rcontext &db = cntx::rcontext::instance();
  auto& f = db.createnode("Film",name,true,db.getUnique("Films"));
  db.getChild(f,"width") = w;
  db.getChild(f,"height") = h;
  db.getChild(f,"outputPath") = path;
}

/* modify film object, if it exists
 * \param w the image width
 * \param h the image height
 * \param path the path for the image file
 */
void modifyFilm(string name, int w, int h, string path) {
  cntx::rcontext &db = cntx::rcontext::instance();
  auto& f = db.getUnique(name);
  if(f.getid().isInvalid()) {
    return;
  }
  db.getChild(f,"width") = w;
  db.getChild(f,"height") = h;
  db.getChild(f,"outputPath") = path;
}



/**
 * add a renderer to the context
 * \param name the renderer name
 * \param adapter the rendering adapter / engine used (ospray,embree,optix,manta)
 * \param schedule the schedule to use for this adapter (image,domain,hybrid)
 */
void addRenderer(string name, int adapter, int schedule, std::string const& Camera, std::string const& Film, bool volume) {
  cntx::rcontext &db = cntx::rcontext::instance();
  auto& s = db.createnode("Scheduler",name,true,db.getUnique("Schedulers"));
  db.getChild(s,"type") = schedule;
  db.getChild(s,"volume") = volume;
  db.getChild(s,"adapter") = adapter;
  db.getChild(s,"camera") = Camera;
  db.getChild(s,"film") = Film;
}

/**
 * modify a renderer in the context, if it exists
 * \param name the renderer name
 * \param adapter the rendering adapter / engine used (ospray,embree,optix,manta)
 * \param schedule the schedule to use for this adapter (image,domain,hybrid)
 */
void modifyRenderer(string name, int adapter, int schedule, std::string const& Camera, std::string const& Film) {
  cntx::rcontext &db = cntx::rcontext::instance();
  auto& s = db.getUnique(name);
  if(s.getid().isInvalid()) return;
  db.getChild(s,"type") = schedule;
  db.getChild(s,"adapter") = adapter;
  db.getChild(s,"camera") = Camera;
  db.getChild(s,"film") = Film;

}

void render(std::string name) {
    gvt::render::gvtRenderer *ren = gvt::render::gvtRenderer::instance();
    ren->render(name);
}

void gvtsync() {
  cntx::rcontext &db = cntx::rcontext::instance();
  db.sync();
}

void writeimage(std::string name, std::string output) {
    gvt::render::gvtRenderer *ren = gvt::render::gvtRenderer::instance();
    ren->WriteImage(output);
}

#ifdef GVT_BUILD_VOLUME
void createVolume(const std::string name, const bool amr) {

  cntx::rcontext &db = cntx::rcontext::instance();
  cntx::node &root = cntx::rcontext::instance().root();
  db.createnode("Volume", name, true, db.getUnique("Data").getid());
  db.getChild(db.getUnique(name), "file") = name;
  db.getChild(db.getUnique(name), "ptr") = std::make_shared<gvt::render::data::primitives::Volume>();
  std::shared_ptr<std::vector<int> > v = std::make_shared<std::vector<int> >();
  v->push_back(db.cntx_comm.rank);
  db.getChild(db.getUnique(name), "Locations") = v; // db.cntx_comm.rank;
  if ( amr ) {
    std::shared_ptr<gvt::render::data::primitives::Volume> vol = getChildByName(db.getUnique(name),"ptr");
    vol->SetAMRTrue();
  }

}

void addVolumeTransferFunctions(const std::string name, const std::string colortfname, const std::string opacitytfname,float low,float high) {
  cntx::rcontext &db = cntx::rcontext::instance();
  std::shared_ptr<gvt::render::data::primitives::Volume> v = getChildByName(db.getUnique(name), "ptr");
  gvt::render::data::primitives::TransferFunction* tf = new gvt::render::data::primitives::TransferFunction();
  tf->load(colortfname,opacitytfname);
  tf->setValueRange(glm::vec2(low,high));
  v->SetTransferFunction(tf);
}

void addVolumeSamples(const std::string name,  float *samples,  int *counts,  float *origin,  float *deltas, float samplingrate, double *bounds ) {
    float dx,dy,dz;
  cntx::rcontext &db = cntx::rcontext::instance();
  std::shared_ptr<gvt::render::data::primitives::Volume> v = getChildByName(db.getUnique(name), "ptr");
  v->SetVoxelType(gvt::render::data::primitives::Volume::FLOAT);
  v->SetSamples(samples);
  v->SetCounts(counts[0],counts[1],counts[2]);
  v->SetOrigin(origin[0],origin[1],origin[2]);
  v->SetDeltas(deltas[0],deltas[1],deltas[2]);
  v->SetSamplingRate(samplingrate);
  std::cerr << " api: bounds " << bounds[0] << "\n" 
  << bounds[1] << "\n"
  << bounds[2] << "\n"
  << bounds[3] << "\n"
  << bounds[4] << "\n"
  << bounds[5] << "\n"<< std::endl;
  glm::vec3 lower(bounds[0],bounds[2],bounds[4]);
  glm::vec3 upper(bounds[1],bounds[3],bounds[5]);
  //glm::vec3 lower(origin[0],origin[1],origin[2]);
  dx = deltas[0]*(float)(counts[0] - 1);
  dy = deltas[1]*(float)(counts[1] - 1);
  dz = deltas[2]*(float)(counts[2] - 1);
  //glm::vec3 upper = lower + glm::vec3(dx,dy,dz);
  v->SetBoundingBox(lower,upper);
  db.getChild(db.getUnique(name), "bbox") = std::make_shared<gvt::render::data::primitives::Box3D>(lower,upper);
  if(v->is_AMR()) {
      v->SetAMRLevels(1); // first level on this call 
      v->SetAMRNumberOfGridsInVolume(0); // addAmrSubgrid increments this. 
      v->SetAMRlng(0,0); // addAmrSubgrid increments this.
      v->SetAMRBounds(bounds); // set the bounds of the level 0 grid
      addAmrSubgrid(name,0,0,samples,counts,origin,deltas);// level0 grid is first in list

  } else {
      v->SetSamples(samples);
  }
}

void addAmrSubgrid(const std::string name, int gridid, int level, float *samples, int *counts, float *origin, float *deltas) {
    cntx::rcontext &db = cntx::rcontext::instance();
    std::shared_ptr<gvt::render::data::primitives::Volume> v = getChildByName(db.getUnique(name), "ptr");
    // now set subgrid
    v->AddAMRGrid(gridid,level,origin,deltas,counts,samples);
}
#endif // GVT_BUILD_VOLUME

} // namespace api
