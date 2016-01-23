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
/**
 * A simple GraviT application that loads some geometry and renders it.
 *
 * This application renders a simple scene of cones and cubes using the GraviT interface.
 * This will run in both single-process and MPI modes. You can alter the work scheduler
 * used by changing line 242.
 *
*/
#include <gvt/render/RenderContext.h>
#include <gvt/render/Types.h>
#include <vector>
#include <algorithm>
#include <set>
#include <gvt/core/mpi/Wrapper.h>
#include <gvt/core/Math.h>
#include <gvt/render/data/Dataset.h>
#include <gvt/render/data/Domains.h>
#include <gvt/render/Schedulers.h>

#ifdef GVT_RENDER_ADAPTER_EMBREE
#include <gvt/render/adapter/embree/Wrapper.h>
#endif

#ifdef GVT_RENDER_ADAPTER_MANTA
#include <gvt/render/adapter/manta/Wrapper.h>
#endif

#ifdef GVT_RENDER_ADAPTER_OPTIX
#include <gvt/render/adapter/optix/Wrapper.h>
#endif

#ifdef GVT_USE_MPE
#include "mpe.h"
#endif
#include <gvt/render/algorithm/Tracers.h>
#include <gvt/render/data/scene/gvtCamera.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/data/Primitives.h>

#include <boost/range/algorithm.hpp>

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <ply.h>

using namespace std;
using namespace gvt::render;
using namespace gvt::core::math;
using namespace gvt::core::mpi;
using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;
using namespace gvt::render::data::primitives;

void test_bvh(gvtPerspectiveCamera &camera);
typedef struct Vertex {
  float x, y, z;
  float nx, ny, nz;
  void *other_props; /* other properties */
} Vertex;

typedef struct Face {
  unsigned char nverts; /* number of vertex indices in list */
  int *verts;           /* vertex index list */
  void *other_props;    /* other properties */
} Face;

PlyProperty vert_props[] = {
  /* list of property information for a vertex */
  { "x", Float32, Float32, offsetof(Vertex, x), 0, 0, 0, 0 },
  { "y", Float32, Float32, offsetof(Vertex, y), 0, 0, 0, 0 },
  { "z", Float32, Float32, offsetof(Vertex, z), 0, 0, 0, 0 },
  { "nx", Float32, Float32, offsetof(Vertex, nx), 0, 0, 0, 0 },
  { "ny", Float32, Float32, offsetof(Vertex, ny), 0, 0, 0, 0 },
  { "nz", Float32, Float32, offsetof(Vertex, nz), 0, 0, 0, 0 },
};

PlyProperty face_props[] = {
  /* list of property information for a face */
  { "vertex_indices", Int32, Int32, offsetof(Face, verts), 1, Uint8, Uint8, offsetof(Face, nverts) },
};
static Vertex **vlist;
static Face **flist;

#define MIN(a, b) ((a < b) ? (a) : (b))
#define MAX(a, b) ((a > b) ? (a) : (b))
int main(int argc, char **argv) {

  // mess I use to open and read the ply file with the c utils I found.
  PlyFile *in_ply;
  Vertex *vert;
  Face *face;
  int elem_count, nfaces, nverts;
  int i, j, k;
  float xmin, ymin, zmin, xmax, ymax, zmax;
  char *elem_name;
  ;
  FILE *myfile;
  char txt[16];
  std::string temp;
  std::string filename, filepath, rootdir;
  rootdir = "/Users/jbarbosa/r/EnzoPlyData/";
  // rootdir = "/work/01197/semeraro/maverick/DAVEDATA/EnzoPlyData/";
  // filename = "/work/01197/semeraro/maverick/DAVEDATA/EnzoPlyData/block0.ply";
  // myfile = fopen(filename.c_str(),"r");
  MPI_Init(&argc, &argv);
  MPI_Pcontrol(0);
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  if (cntxt == NULL) {
    std::cout << "Something went wrong initializing the context" << std::endl;
    exit(0);
  }
  gvt::core::DBNodeH root = cntxt->getRootNode();
  gvt::core::DBNodeH dataNodes = cntxt->createNodeFromType("Data", "Data", root.UUID());
  gvt::core::DBNodeH instNodes = cntxt->createNodeFromType("Instances", "Instances", root.UUID());

  // Enzo isosurface...
  for (k = 0; k < 8; k++) {
    sprintf(txt, "%d", k);
    filename = "block";
    filename += txt;
    gvt::core::DBNodeH EnzoMeshNode = cntxt->createNodeFromType("Mesh", filename.c_str(), dataNodes.UUID());
    // read in some ply data and get ready to load it into the mesh
    // filepath = rootdir + "block" + std::string(txt) + ".ply";
    filepath = rootdir + filename + ".ply";
    myfile = fopen(filepath.c_str(), "r");
    in_ply = read_ply(myfile);
    for (i = 0; i < in_ply->num_elem_types; i++) {
      elem_name = setup_element_read_ply(in_ply, i, &elem_count);
      temp = elem_name;
      if (temp == "vertex") {
        vlist = (Vertex **)malloc(sizeof(Vertex *) * elem_count);
        nverts = elem_count;
        setup_property_ply(in_ply, &vert_props[0]);
        setup_property_ply(in_ply, &vert_props[1]);
        setup_property_ply(in_ply, &vert_props[2]);
        for (j = 0; j < elem_count; j++) {
          vlist[j] = (Vertex *)malloc(sizeof(Vertex));
          get_element_ply(in_ply, (void *)vlist[j]);
        }
      } else if (temp == "face") {
        flist = (Face **)malloc(sizeof(Face *) * elem_count);
        nfaces = elem_count;
        setup_property_ply(in_ply, &face_props[0]);
        for (j = 0; j < elem_count; j++) {
          flist[j] = (Face *)malloc(sizeof(Face));
          get_element_ply(in_ply, (void *)flist[j]);
        }
      }
    }
    close_ply(in_ply);
    // smoosh data into the mesh object
    {
      Mesh *mesh = new Mesh(new Lambert(Vector4f(1.0, 1.0, 1.0, 1.0)));
      vert = vlist[0];
      xmin = vert->x;
      ymin = vert->y;
      zmin = vert->z;
      xmax = vert->x;
      ymax = vert->y;
      zmax = vert->z;

      for (i = 0; i < nverts; i++) {
        vert = vlist[i];
        xmin = MIN(vert->x, xmin);
        ymin = MIN(vert->y, ymin);
        zmin = MIN(vert->z, zmin);
        xmax = MAX(vert->x, xmax);
        ymax = MAX(vert->y, ymax);
        zmax = MAX(vert->z, zmax);
        mesh->addVertex(Point4f(vert->x, vert->y, vert->z, 1.0));
      }
      Point4f lower(xmin, ymin, zmin);
      Point4f upper(xmax, ymax, zmax);
      Box3D *meshbbox = new gvt::render::data::primitives::Box3D(lower, upper);
      // add faces to mesh
      for (i = 0; i < nfaces; i++) {
        face = flist[i];
        mesh->addFace(face->verts[0] + 1, face->verts[1] + 1, face->verts[2] + 1);
      }
      mesh->generateNormals();
      // add Enzo mesh to the database
      // EnzoMeshNode["file"] = string("/work/01197/semeraro/maverick/DAVEDATA/EnzoPlyDATA/Block0.ply");
      EnzoMeshNode["file"] = string(filepath);
      EnzoMeshNode["bbox"] = (unsigned long long)meshbbox;
      EnzoMeshNode["ptr"] = (unsigned long long)mesh;
    }
    // add instance
    gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance", "inst", instNodes.UUID());
    gvt::core::DBNodeH meshNode = EnzoMeshNode;
    Box3D *mbox = (Box3D *)meshNode["bbox"].value().toULongLong();
    instnode["id"] = k;
    instnode["meshRef"] = meshNode.UUID();
    auto m = new gvt::core::math::AffineTransformMatrix<float>(true);
    auto minv = new gvt::core::math::AffineTransformMatrix<float>(true);
    auto normi = new gvt::core::math::Matrix3f();
    instnode["mat"] = (unsigned long long)m;
    *minv = m->inverse();
    instnode["matInv"] = (unsigned long long)minv;
    *normi = m->upper33().inverse().transpose();
    instnode["normi"] = (unsigned long long)normi;
    auto il = (*m) * mbox->bounds[0];
    auto ih = (*m) * mbox->bounds[1];
    Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
    instnode["bbox"] = (unsigned long long)ibox;
    instnode["centroid"] = ibox->centroid();
  }

  // add lights, camera, and film to the database
  gvt::core::DBNodeH lightNodes = cntxt->createNodeFromType("Lights", "Lights", root.UUID());
  gvt::core::DBNodeH lightNode = cntxt->createNodeFromType("PointLight", "conelight", lightNodes.UUID());
  lightNode["position"] = Vector4f(512.0, 512.0, 2048.0, 0.0);
  lightNode["color"] = Vector4f(1.0, 1.0, 1.0, 0.0);
  // camera
  gvt::core::DBNodeH camNode = cntxt->createNodeFromType("Camera", "conecam", root.UUID());
  camNode["eyePoint"] = Point4f(512.0, 512.0, 4096.0, 1.0);
  camNode["focus"] = Point4f(512.0, 512.0, 0.0, 1.0);
  camNode["upVector"] = Vector4f(0.0, 1.0, 0.0, 0.0);
  camNode["fov"] = (float)(25.0 * M_PI / 180.0);
  // film
  gvt::core::DBNodeH filmNode = cntxt->createNodeFromType("Film", "conefilm", root.UUID());
  filmNode["width"] = 2000;
  filmNode["height"] = 2000;

  gvt::core::DBNodeH schedNode = cntxt->createNodeFromType("Schedule", "Enzosched", root.UUID());
  schedNode["type"] = gvt::render::scheduler::Image;
// schedNode["type"] = gvt::render::scheduler::Domain;

#ifdef GVT_RENDER_ADAPTER_EMBREE
  int adapterType = gvt::render::adapter::Embree;
#elif GVT_RENDER_ADAPTER_MANTA
  int adapterType = gvt::render::adapter::Manta;
#elif GVT_RENDER_ADAPTER_OPTIX
  int adapterType = gvt::render::adapter::Optix;
#elif
  GVT_DEBUG(DBG_ALWAYS, "ERROR: missing valid adapter");
#endif

  schedNode["adapter"] = gvt::render::adapter::Embree;

  // end db setup

  // use db to create structs needed by system

  // setup gvtCamera from database entries
  gvtPerspectiveCamera mycamera;
  Point4f cameraposition = camNode["eyePoint"].value().toPoint4f();
  Point4f focus = camNode["focus"].value().toPoint4f();
  float fov = camNode["fov"].value().toFloat();
  Vector4f up = camNode["upVector"].value().toVector4f();
  mycamera.lookAt(cameraposition, focus, up);
  mycamera.setFOV(fov);
  mycamera.setFilmsize(filmNode["width"].value().toInteger(), filmNode["height"].value().toInteger());

#ifdef GVT_USE_MPE
  MPE_Log_event(readend, 0, NULL);
#endif
  // setup image from database sizes
  Image myimage(mycamera.getFilmSizeWidth(), mycamera.getFilmSizeHeight(), "enzo");

  mycamera.AllocateCameraRays();
  mycamera.generateRays();

  int schedType = root["Schedule"]["type"].value().toInteger();
  switch (schedType) {
  case gvt::render::scheduler::Image: {
    std::cout << "starting image scheduler" << std::endl;
    for (int z = 0; z < 1; z++) {
      mycamera.AllocateCameraRays();
      mycamera.generateRays();
      gvt::render::algorithm::Tracer<ImageScheduler>(mycamera.rays, myimage)();
    }
    break;
  }
  case gvt::render::scheduler::Domain: {
    std::cout << "starting domain scheduler" << std::endl;
#ifdef GVT_USE_MPE
    MPE_Log_event(renderstart, 0, NULL);
#endif
    gvt::render::algorithm::Tracer<DomainScheduler>(mycamera.rays, myimage)();
#ifdef GVT_USE_MPE
    MPE_Log_event(renderend, 0, NULL);
#endif
    break;
  }
  default: {
    std::cout << "unknown schedule type provided: " << schedType << std::endl;
    break;
  }
  }

  myimage.Write();
#ifdef GVT_USE_MPE
  MPE_Log_sync_clocks();
// MPE_Finish_log("gvtSimplelog");
#endif
  if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();
}

// bvh intersection list test
void test_bvh(gvtPerspectiveCamera &mycamera) {
  gvt::core::DBNodeH root = gvt::render::RenderContext::instance()->getRootNode();

  cout << "\n-- bvh test --" << endl;

  auto ilist = root["Instances"].getChildren();
  auto bvh = new gvt::render::data::accel::BVH(ilist);

  // list of rays to test
  std::vector<gvt::render::actor::Ray> rays;
  rays.push_back(mycamera.rays[100 * 512 + 100]);
  rays.push_back(mycamera.rays[182 * 512 + 182]);
  rays.push_back(mycamera.rays[256 * 512 + 256]);
  auto dir =
      (gvt::core::math::Vector4f(0.0, 0.0, 0.0, 0.0) - gvt::core::math::Vector4f(1.0, 1.0, 1.0, 0.0)).normalize();
  rays.push_back(gvt::render::actor::Ray(gvt::core::math::Point4f(1.0, 1.0, 1.0, 1.0), dir));
  rays.push_back(mycamera.rays[300 * 512 + 300]);
  rays.push_back(mycamera.rays[400 * 512 + 400]);
  rays.push_back(mycamera.rays[470 * 512 + 470]);
  rays.push_back(gvt::render::actor::Ray(gvt::core::math::Point4f(0.0, 0.0, 1.0, 1.0),
                                         gvt::core::math::Vector4f(0.0, 0.0, -1.0, 1.0)));
  rays.push_back(mycamera.rays[144231]);

  // test rays and print out which instances were hit
  for (size_t z = 0; z < rays.size(); z++) {
    gvt::render::actor::Ray &r = rays[z];
    cout << "bvh: r[" << z << "]: " << r << endl;

    gvt::render::actor::isecDomList &isect = r.domains;
    bvh->intersect(r, isect);
    boost::sort(isect);
    cout << "bvh: r[" << z << "]: isect[" << isect.size() << "]: ";
    for (auto i : isect) {
      cout << i.domain << " ";
    }
    cout << endl;
  }

#if 0
    cout << "- check all rays" << endl;
    for(int z=0; z<mycamera.rays.size(); z++) {
        gvt::render::actor::Ray &r = mycamera.rays[z];

        gvt::render::actor::isecDomList& isect = r.domains;
        bvh->intersect(r, isect);
        boost::sort(isect);

        if(isect.size() > 1) {
            cout << "bvh: r[" << z << "]: " << r << endl;
            cout << "bvh: r[" << z << "]: isect[" << isect.size() << "]: ";
            for(auto i : isect) { cout << i.domain << " "; }
            cout << endl;
        }
    }
#endif

  cout << "--------------\n\n" << endl;
}
