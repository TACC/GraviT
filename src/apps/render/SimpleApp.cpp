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
#include <algorithm>
#include <gvt/core/Math.h>
#include <gvt/core/Variant.h>
#include <gvt/core/mpi/Wrapper.h>
#include <gvt/render/RenderContext.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/Types.h>
#include <gvt/render/data/Domains.h>
#include <set>
#include <vector>

#include <tbb/task_scheduler_init.h>
#include <thread>

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
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/data/scene/gvtCamera.h>

#include <boost/range/algorithm.hpp>

#include <iostream>

#ifdef __USE_TAU
#include <TAU.h>
#endif

#include "ParseCommandLine.h"

using namespace std;
using namespace gvt::render;

using namespace gvt::core::mpi;
using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;
using namespace gvt::render::data::primitives;

void test_bvh(gvtPerspectiveCamera &camera);

int main(int argc, char **argv) {

  std::cout << "Ray :" << sizeof(gvt::render::actor::Ray) << std::endl;

  ParseCommandLine cmd("gvtSimple");

  cmd.addoption("wsize", ParseCommandLine::INT, "Window size", 2);
  cmd.addoption("eye", ParseCommandLine::FLOAT, "Camera position", 3);
  cmd.addoption("look", ParseCommandLine::FLOAT, "Camera look at", 3);
  cmd.addoption("lpos", ParseCommandLine::FLOAT, "Light position", 3);
  cmd.addoption("lcolor", ParseCommandLine::FLOAT, "Light color", 3);
  cmd.addoption("image", ParseCommandLine::NONE, "Use embeded scene", 0);
  cmd.addoption("domain", ParseCommandLine::NONE, "Use embeded scene", 0);
  cmd.addoption("threads", ParseCommandLine::INT, "Number of threads to use (default number cores + ht)", 1);
  cmd.addoption("output", ParseCommandLine::PATH, "Output Image Path", 1);
  cmd.addconflict("image", "domain");

  cmd.parse(argc, argv);

  if (!cmd.isSet("threads")) {
    tbb::task_scheduler_init init(std::thread::hardware_concurrency());
  } else {
    tbb::task_scheduler_init init(cmd.get<int>("threads"));
  }

  MPI_Init(&argc, &argv);
  MPI_Pcontrol(0);
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef GVT_USE_MPE
  // MPE_Init_log();
  int readstart, readend;
  int renderstart, renderend;
  MPE_Log_get_state_eventIDs(&readstart, &readend);
  MPE_Log_get_state_eventIDs(&renderstart, &renderend);
  if (rank == 0) {
    MPE_Describe_state(readstart, readend, "Initialize context state", "red");
    MPE_Describe_state(renderstart, renderend, "Render", "yellow");
  }
  MPI_Pcontrol(1);
  MPE_Log_event(readstart, 0, NULL);
#endif
  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  if (cntxt == NULL) {
    std::cout << "Something went wrong initializing the context" << std::endl;
    exit(0);
  }

  gvt::core::DBNodeH root = cntxt->getRootNode();

  // mix of cones and cubes

  if (rank == 0) {
	 gvt::core::DBNodeH dataNodes = cntxt->addToSync(cntxt->createNodeFromType("Data", "Data", root.UUID()));
     cntxt->addToSync(cntxt->createNodeFromType("Mesh", "conemesh", dataNodes.UUID()));
     cntxt->addToSync(cntxt->createNodeFromType("Mesh", "cubemesh", dataNodes.UUID()));
     cntxt->addToSync(cntxt->createNodeFromType("Instances", "Instances", root.UUID()));
   }

  cntxt->syncContext();

  gvt::core::DBNodeH dataNodes = root["Data"];
  gvt::core::DBNodeH instNodes = root["Instances"];

  gvt::core::DBNodeH coneMeshNode = dataNodes.getChildren()[0];
  gvt::core::DBNodeH cubeMeshNode =  dataNodes.getChildren()[1];

  {
    Material* m = new Material();
    m->type = LAMBERT;
    //m->type = EMBREE_MATERIAL_MATTE;
    m->kd = glm::vec3(1.0,1.0, 1.0);
    m->ks = glm::vec3(1.0,1.0,1.0);
    m->alpha = 0.5;

    //m->type = EMBREE_MATERIAL_METAL;
    //copper metal
    m->eta = glm::vec3(.19,1.45, 1.50);
    m->k = glm::vec3(3.06,2.40, 1.88);
    m->roughness = 0.05;

    Mesh *mesh = new Mesh(m);
    int numPoints = 7;
    glm::vec3 points[6];
    points[0] = glm::vec3(0.5, 0.0, 0.0);
    points[1] = glm::vec3(-0.5, 0.5, 0.0);
    points[2] = glm::vec3(-0.5, 0.25, 0.433013);
    points[3] = glm::vec3(-0.5, -0.25, 0.43013);
    points[4] = glm::vec3(-0.5, -0.5, 0.0);
    points[5] = glm::vec3(-0.5, -0.25, -0.433013);
    points[6] = glm::vec3(-0.5, 0.25, -0.433013);

    for (int i = 0; i < numPoints; i++) {
      mesh->addVertex(points[i]);
    }
    mesh->addFace(1, 2, 3);
    mesh->addFace(1, 3, 4);
    mesh->addFace(1, 4, 5);
    mesh->addFace(1, 5, 6);
    mesh->addFace(1, 6, 7);
    mesh->addFace(1, 7, 2);
    mesh->generateNormals();

    // calculate bbox
    glm::vec3 lower = points[0], upper = points[0];
    for (int i = 1; i < numPoints; i++) {
      for (int j = 0; j < 3; j++) {
        lower[j] = (lower[j] < points[i][j]) ? lower[j] : points[i][j];
        upper[j] = (upper[j] > points[i][j]) ? upper[j] : points[i][j];
      }
    }
    Box3D *meshbbox = new gvt::render::data::primitives::Box3D(lower, upper);

    // add cone mesh to the database
    gvt::core::Variant meshvariant(mesh);
//    std::cout << "meshvariant " << meshvariant << std::endl;
    coneMeshNode["file"] = string("/fake/path/to/cone");
    coneMeshNode["bbox"] = (unsigned long long)meshbbox;
    coneMeshNode["ptr"] = (unsigned long long)mesh;

	gvt::core::DBNodeH loc = cntxt->createNode("rank", rank);
	coneMeshNode["Locations"] += loc;

	cntxt->addToSync(coneMeshNode);
  }

  {

    Material* m = new Material();
    m->type = LAMBERT;
    //m->type = EMBREE_MATERIAL_MATTE;
    m->kd = glm::vec3(1.0,1.0, 1.0);
    m->ks = glm::vec3(1.0,1.0,1.0);
    m->alpha = 0.5;

    //m->type = EMBREE_MATERIAL_METAL;
    //copper metal
    m->eta = glm::vec3(.19,1.45, 1.50);
    m->k = glm::vec3(3.06,2.40, 1.88);
    m->roughness = 0.05;

    Mesh *mesh = new Mesh(m);

    int numPoints = 24;
    glm::vec3 points[24];
    points[0] = glm::vec3(-0.5, -0.5, 0.5);
    points[1] = glm::vec3(0.5, -0.5, 0.5);
    points[2] = glm::vec3(0.5, 0.5, 0.5);
    points[3] = glm::vec3(-0.5, 0.5, 0.5);
    points[4] = glm::vec3(-0.5, -0.5, -0.5);
    points[5] = glm::vec3(0.5, -0.5, -0.5);
    points[6] = glm::vec3(0.5, 0.5, -0.5);
    points[7] = glm::vec3(-0.5, 0.5, -0.5);

    points[8] = glm::vec3(0.5, 0.5, 0.5);
    points[9] = glm::vec3(-0.5, 0.5, 0.5);
    points[10] = glm::vec3(0.5, 0.5, -0.5);
    points[11] = glm::vec3(-0.5, 0.5, -0.5);

    points[12] = glm::vec3(-0.5, -0.5, 0.5);
    points[13] = glm::vec3(0.5, -0.5, 0.5);
    points[14] = glm::vec3(-0.5, -0.5, -0.5);
    points[15] = glm::vec3(0.5, -0.5, -0.5);

    points[16] = glm::vec3(0.5, -0.5, 0.5);
    points[17] = glm::vec3(0.5, 0.5, 0.5);
    points[18] = glm::vec3(0.5, -0.5, -0.5);
    points[19] = glm::vec3(0.5, 0.5, -0.5);

    points[20] = glm::vec3(-0.5, -0.5, 0.5);
    points[21] = glm::vec3(-0.5, 0.5, 0.5);
    points[22] = glm::vec3(-0.5, -0.5, -0.5);
    points[23] = glm::vec3(-0.5, 0.5, -0.5);

    for (int i = 0; i < numPoints; i++) {
      mesh->addVertex(points[i]);
    }
    // faces are 1 indexed
    mesh->addFace(1, 2, 3);
    mesh->addFace(1, 3, 4);

    mesh->addFace(17, 19, 20);
    mesh->addFace(17, 20, 18);

    mesh->addFace(6, 5, 8);
    mesh->addFace(6, 8, 7);

    mesh->addFace(23, 21, 22);
    mesh->addFace(23, 22, 24);

    mesh->addFace(10, 9, 11);
    mesh->addFace(10, 11, 12);

    mesh->addFace(13, 15, 16);
    mesh->addFace(13, 16, 14);

    mesh->generateNormals();

    // calculate bbox
    glm::vec3 lower = points[0], upper = points[0];
    for (int i = 1; i < numPoints; i++) {
      for (int j = 0; j < 3; j++) {
        lower[j] = (lower[j] < points[i][j]) ? lower[j] : points[i][j];
        upper[j] = (upper[j] > points[i][j]) ? upper[j] : points[i][j];
      }
    }
    Box3D *meshbbox = new gvt::render::data::primitives::Box3D(lower, upper);

    // add cube mesh to the database
    cubeMeshNode["file"] = string("/fake/path/to/cube");
    cubeMeshNode["bbox"] = (unsigned long long)meshbbox;
    cubeMeshNode["ptr"] = (unsigned long long)mesh;

	gvt::core::DBNodeH loc = cntxt->createNode("rank", rank);
	cubeMeshNode["Locations"] += loc;

    cntxt->addToSync(cubeMeshNode);

  }

  cntxt->syncContext();

  if (rank ==0 ) {
  // create a NxM grid of alternating cones / cubes, offset using i and j
  int instId = 0;
  int ii[2] = { -2, 3 }; // i range
  int jj[2] = { -2, 3 }; // j range
  for (int i = ii[0]; i < ii[1]; i++) {
    for (int j = jj[0]; j < jj[1]; j++) {
      gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance", "inst", instNodes.UUID());
      // gvt::core::DBNodeH meshNode = (instId % 2) ? coneMeshNode : cubeMeshNode;
      gvt::core::DBNodeH meshNode = (instId % 2) ? cubeMeshNode : coneMeshNode;
      Box3D *mbox = (Box3D *)meshNode["bbox"].value().toULongLong();

      instnode["id"] = instId++;
      instnode["meshRef"] = meshNode.UUID();

      auto m = new glm::mat4(1.f);
      auto minv = new glm::mat4(1.f);
      auto normi = new glm::mat3(1.f);
      //*m *glm::mat4::createTranslation(0.0, i * 0.5, j * 0.5);
      *m = glm::translate(*m, glm::vec3(0.0, i * 0.5, j * 0.5));
      //*m = *m * glm::mat4::createScale(0.4, 0.4, 0.4);
      *m = glm::scale(*m, glm::vec3(0.4, 0.4, 0.4));

      instnode["mat"] = (unsigned long long)m;
      *minv = glm::inverse(*m);
      instnode["matInv"] = (unsigned long long)minv;
      *normi = glm::transpose(glm::inverse(glm::mat3(*m)));
      instnode["normi"] = (unsigned long long)normi;
      auto il = glm::vec3((*m) * glm::vec4(mbox->bounds_min, 1.f));
      auto ih = glm::vec3((*m) * glm::vec4(mbox->bounds_max, 1.f));
      Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
      instnode["bbox"] = (unsigned long long)ibox;
      instnode["centroid"] = ibox->centroid();

	  cntxt->addToSync(instnode);

    }
  }
  }

  cntxt->syncContext();


  // add lights, camera, and film to the database
  gvt::core::DBNodeH lightNodes = cntxt->createNodeFromType("Lights", "Lights", root.UUID());
 #if 1
  gvt::core::DBNodeH lightNode = cntxt->createNodeFromType("PointLight", "conelight", lightNodes.UUID());
  lightNode["position"] = glm::vec3(1.0, 0.0, -1.0);
  lightNode["color"] = glm::vec3(1.0, 1.0, 1.0);
#else
  gvt::core::DBNodeH lightNode = cntxt->createNodeFromType(
              "AreaLight", "AreaLight", lightNodes.UUID());

  lightNode["position"] = glm::vec3(1.0, 0.0, 0.0);
  lightNode["normal"] = glm::vec3(-1.0, 0.0, 0.0);
  lightNode["width"] = 2.f;
  lightNode["height"] = 2.f;
  lightNode["color"] = glm::vec3(1.0, 1.0, 1.0);
#endif
  // second light just for fun
  // gvt::core::DBNodeH lN2 = cntxt->createNodeFromType("PointLight", "conelight", lightNodes.UUID());
  // lN2["position"] = glm::vec3(2.0, 2.0, 2.0, 0.0);
  // lN2["color"] = glm::vec3(0.0, 0.0, 0.0, 0.0);

  gvt::core::DBNodeH camNode = cntxt->createNodeFromType("Camera", "conecam", root.UUID());
  camNode["eyePoint"] = glm::vec3(4.0, 0.0, 0.0);
  camNode["focus"] = glm::vec3(0.0, 0.0, 0.0);
  camNode["upVector"] = glm::vec3(0.0, 1.0, 0.0);
  camNode["fov"] = (float)(45.0 * M_PI / 180.0);
  camNode["rayMaxDepth"] = (int)1;
  camNode["raySamples"] = (int)1;
  camNode["jitterWindowSize"] = (float)0.5;

  gvt::core::DBNodeH filmNode = cntxt->createNodeFromType("Film", "conefilm", root.UUID());
  filmNode["width"] = 512;
  filmNode["height"] = 512;
  filmNode["outputPath"] = (std::string)"simple";

  if (cmd.isSet("lpos")) {
    std::vector<float> pos = cmd.getValue<float>("lpos");
    lightNode["position"] = glm::vec3(pos[0], pos[1], pos[2]);
  }
  if (cmd.isSet("lcolor")) {
    std::vector<float> color = cmd.getValue<float>("lcolor");
    lightNode["color"] = glm::vec3(color[0], color[1], color[2]);
  }

  if (cmd.isSet("eye")) {
    std::vector<float> eye = cmd.getValue<float>("eye");
    camNode["eyePoint"] = glm::vec3(eye[0], eye[1], eye[2]);
  }

  if (cmd.isSet("look")) {
    std::vector<float> eye = cmd.getValue<float>("look");
    camNode["focus"] = glm::vec3(eye[0], eye[1], eye[2]);
  }
  if (cmd.isSet("wsize")) {
    std::vector<int> wsize = cmd.getValue<int>("wsize");
    filmNode["width"] = wsize[0];
    filmNode["height"] = wsize[1];
  }
  if (cmd.isSet("output"))
  {
    std::vector<std::string> output = cmd.getValue<std::string>("output");
    filmNode["outputPath"] = output[0];
  }


  gvt::core::DBNodeH schedNode = cntxt->createNodeFromType("Schedule", "Enzosched", root.UUID());
  if (cmd.isSet("domain"))
    schedNode["type"] = gvt::render::scheduler::Domain;
  else
    schedNode["type"] = gvt::render::scheduler::Image;

#ifdef GVT_RENDER_ADAPTER_EMBREE
  int adapterType = gvt::render::adapter::Embree;
#elif GVT_RENDER_ADAPTER_MANTA
  int adapterType = gvt::render::adapter::Manta;
#elif GVT_RENDER_ADAPTER_OPTIX
  int adapterType = gvt::render::adapter::Optix;
#else
  GVT_DEBUG(DBG_ALWAYS, "ERROR: missing valid adapter");
#endif

  schedNode["adapter"] = adapterType;

  // end db setup

  // cntxt->database()->printTree(root.UUID(), 10, std::cout);

  // use db to create structs needed by system

  // setup gvtCamera from database entries
  gvtPerspectiveCamera mycamera;
  glm::vec3 cameraposition = camNode["eyePoint"].value().tovec3();
  glm::vec3 focus = camNode["focus"].value().tovec3();
  float fov = camNode["fov"].value().toFloat();
  glm::vec3 up = camNode["upVector"].value().tovec3();

  int rayMaxDepth = camNode["rayMaxDepth"].value().toInteger();
  int raySamples = camNode["raySamples"].value().toInteger();
  float jitterWindowSize = camNode["jitterWindowSize"].value().toFloat();

  mycamera.setMaxDepth(rayMaxDepth);
  mycamera.setSamples(raySamples);
  mycamera.setJitterWindowSize(jitterWindowSize);
  mycamera.lookAt(cameraposition, focus, up);
  mycamera.setFOV(fov);
  mycamera.setFilmsize(filmNode["width"].value().toInteger(), filmNode["height"].value().toInteger());

#ifdef GVT_USE_MPE
  MPE_Log_event(readend, 0, NULL);
#endif
  // setup image from database sizes
  Image myimage(mycamera.getFilmSizeWidth(), mycamera.getFilmSizeHeight(), filmNode["outputPath"].value().toString());

  mycamera.AllocateCameraRays();
  mycamera.generateRays();

  int schedType = root["Schedule"]["type"].value().toInteger();
  switch (schedType) {
  case gvt::render::scheduler::Image: {
 //   std::cout << "starting image scheduler" << std::endl;
  //  std::cout << "ligthpos " << lightNode["position"].value().tovec3() << std::endl;
    gvt::render::algorithm::Tracer<ImageScheduler> tracer(mycamera.rays, myimage);
    for (int z = 0; z < 10; z++) {
      mycamera.AllocateCameraRays();
      mycamera.generateRays();
      myimage.clear();
      tracer();
    }
    break;
  }
  case gvt::render::scheduler::Domain: {
    //std::cout << "starting domain scheduler" << std::endl;
#ifdef GVT_USE_MPE
    MPE_Log_event(renderstart, 0, NULL);
#endif
    // gvt::render::algorithm::Tracer<DomainScheduler>(mycamera.rays, myimage)();
    gvt::render::algorithm::Tracer<DomainScheduler> tracer(mycamera.rays, myimage);
    for (int z = 0; z < 10; z++) {
      mycamera.AllocateCameraRays();
      mycamera.generateRays();
      myimage.clear();
      tracer();
    }
    break;
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
//
// // bvh intersection list test
// void test_bvh(gvtPerspectiveCamera &mycamera) {
//   gvt::core::DBNodeH root = gvt::render::RenderContext::instance()->getRootNode();
//
//   cout << "\n-- bvh test --" << endl;
//
//   auto ilist = root["Instances"].getChildren();
//   auto bvh = new gvt::render::data::accel::BVH(ilist);
//
//   // list of rays to test
//   std::vector<gvt::render::actor::Ray> rays;
//   rays.push_back(mycamera.rays[100 * 512 + 100]);
//   rays.push_back(mycamera.rays[182 * 512 + 182]);
//   rays.push_back(mycamera.rays[256 * 512 + 256]);
//   auto dir = glm::normalize(glm::vec3(0.0, 0.0, 0.0) - glm::vec3(1.0, 1.0, 1.0));
//   rays.push_back(gvt::render::actor::Ray(glm::vec3(1.0, 1.0, 1.0), dir));
//   rays.push_back(mycamera.rays[300 * 512 + 300]);
//   rays.push_back(mycamera.rays[400 * 512 + 400]);
//   rays.push_back(mycamera.rays[470 * 512 + 470]);
//   rays.push_back(gvt::render::actor::Ray(glm::vec3(0.0, 0.0, 1.0), glm::vec3(0.0, 0.0, -1.0)));
//   rays.push_back(mycamera.rays[144231]);
//
//   // test rays and print out which instances were hit
//   for (size_t z = 0; z < rays.size(); z++) {
//     gvt::render::actor::Ray &r = rays[z];
//     cout << "bvh: r[" << z << "]: " << r << endl;
//
//     gvt::render::actor::isecDomList &isect = r.domains;
//     bvh->intersect(r, isect);
//     std::sort(isect.begin(), isect.end());
//     cout << "bvh: r[" << z << "]: isect[" << isect.size() << "]: ";
//     for (auto i : isect) {
//       cout << i.domain << " ";
//     }
//     cout << endl;
//   }
//
// #if 0
//     cout << "- check all rays" << endl;
//     for(int z=0; z<mycamera.rays.size(); z++) {
//         gvt::render::actor::Ray &r = mycamera.rays[z];
//
//         gvt::render::actor::isecDomList& isect = r.domains;
//         bvh->intersect(r, isect);
//         std::sort(isect);
//
//         if(isect.size() > 1) {
//             cout << "bvh: r[" << z << "]: " << r << endl;
//             cout << "bvh: r[" << z << "]: isect[" << isect.size() << "]: ";
//             for(auto i : isect) { cout << i.domain << " "; }
//             cout << endl;
//         }
//     }
// #endif
//
//   cout << "--------------\n\n" << endl;
// }
