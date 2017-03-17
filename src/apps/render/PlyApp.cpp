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
 * This will run in both single-process and MPI modes.
 *
*/
#include <algorithm>
#include <gvt/core/Math.h>
#include <gvt/render/RenderContext.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/Types.h>
#include <gvt/render/data/Domains.h>
#include <set>
#include <vector>

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

#include <gvt/render/algorithm/Tracers.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/data/scene/gvtCamera.h>

#include "ParseCommandLine.h"
#include <boost/range/algorithm.hpp>
#include <gvt/render/data/reader/PlyReader.h>

using namespace std;
using namespace gvt::render;
using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;
using namespace gvt::render::data::primitives;

#include <tbb/task_scheduler_observer.h>
class concurrency_tracker : public tbb::task_scheduler_observer {
  tbb::atomic<int> num_threads;

public:
  concurrency_tracker() : num_threads() { observe(true); }
  /*override*/ void on_scheduler_entry(bool) { ++num_threads; }
  /*override*/ void on_scheduler_exit(bool) { --num_threads; }

  int get_concurrency() { return num_threads; }
};

// Used for testing purposes where it specifies the number of ply blocks read by each mpi
//#define DOMAIN_PER_NODE 2

int main(int argc, char **argv) {
 {
  gvt::core::time::timer t_skip(true, "To skip");

  ParseCommandLine cmd("gvtPly");

  cmd.addoption("wsize", ParseCommandLine::INT, "Window size", 2);
  cmd.addoption("eye", ParseCommandLine::FLOAT, "Camera position", 3);
  cmd.addoption("look", ParseCommandLine::FLOAT, "Camera look at", 3);
  cmd.addoption("file", ParseCommandLine::PATH | ParseCommandLine::REQUIRED, "File path");
  cmd.addoption("image", ParseCommandLine::NONE, "Use embeded scene", 0);
  cmd.addoption("domain", ParseCommandLine::NONE, "Use embeded scene", 0);
  cmd.addoption("threads", ParseCommandLine::INT, "Number of threads to use (default number cores + ht)", 1);
  cmd.addoption("embree", ParseCommandLine::NONE, "Embree Adapter Type", 0);
  cmd.addoption("manta", ParseCommandLine::NONE, "Manta Adapter Type", 0);
  cmd.addoption("optix", ParseCommandLine::NONE, "Optix Adapter Type", 0);

  cmd.addconflict("image", "domain");
  cmd.addconflict("embree", "manta");
  cmd.addconflict("embree", "optix");
  cmd.addconflict("manta", "optix");

  cmd.parse(argc, argv);

  tbb::task_scheduler_init *init;
  if (!cmd.isSet("threads")) {
    init = new tbb::task_scheduler_init(std::thread::hardware_concurrency());
    std::cout << "Initialized GraviT with " << std::thread::hardware_concurrency() << " threads..." << std::endl;
  } else {
    init = new tbb::task_scheduler_init(cmd.get<int>("threads"));
    std::cout << "Initialized GraviT with " << cmd.get<int>("threads") << " threads..." << std::endl;
  }

  //  concurrency_tracker tracker;
  //  tracker.observe(true);

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
  root += cntxt->createNode(
      "threads", cmd.isSet("threads") ? (int)cmd.get<int>("threads") : (int)std::thread::hardware_concurrency());

  // A single mpi node will create db nodes and then broadcast them
  if (MPI::COMM_WORLD.Get_rank() == 0) {
    cntxt->addToSync(cntxt->createNodeFromType("Data", "Data", root.UUID()));
    cntxt->addToSync(cntxt->createNodeFromType("Instances", "Instances", root.UUID()));
  }

  cntxt->syncContext();

  gvt::core::DBNodeH dataNodes = root["Data"];
  gvt::core::DBNodeH instNodes = root["Instances"];

  gvt::render::data::domain::reader::PlyReader plyReader(cmd.get<std::string>("file"));

  // context has the location information of the domain, so for simplicity only one mpi will create the instances
  if (MPI::COMM_WORLD.Get_rank() == 0) {
    for (int k = 0; k < plyReader.getMeshes().size(); k++) {

      // add instance
      gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance", "inst", instNodes.UUID());
      gvt::core::DBNodeH meshNode = dataNodes.getChildren()[k];
      Box3D *mbox = (Box3D *)meshNode["bbox"].value().toULongLong();
      instnode["id"] = k;
      instnode["meshRef"] = meshNode.UUID();
      auto m = new glm::mat4(1.f);
      auto minv = new glm::mat4(1.f);
      auto normi = new glm::mat3(1.f);
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

  cntxt->syncContext();

  // add lights, camera, and film to the database
  gvt::core::DBNodeH lightNodes = cntxt->createNodeFromType("Lights", "Lights", root.UUID());
  gvt::core::DBNodeH lightNode = cntxt->createNodeFromType("PointLight", "conelight", lightNodes.UUID());
  lightNode["position"] = glm::vec3(512.0, 512.0, 2048.0);
  lightNode["color"] = glm::vec3(1000.0, 1000.0, 1000.0);
  // camera
  gvt::core::DBNodeH camNode = cntxt->createNodeFromType("Camera", "conecam", root.UUID());
  camNode["eyePoint"] = glm::vec3(512.0, 512.0, 4096.0);
  camNode["focus"] = glm::vec3(512.0, 512.0, 0.0);
  camNode["upVector"] = glm::vec3(0.0, 1.0, 0.0);
  camNode["fov"] = (float)(25.0 * M_PI / 180.0);
  camNode["rayMaxDepth"] = (int)1;
  camNode["raySamples"] = (int)1;
  // film
  gvt::core::DBNodeH filmNode = cntxt->createNodeFromType("Film", "conefilm", root.UUID());
  filmNode["width"] = 1900;
  filmNode["height"] = 1080;

  if (cmd.isSet("eye")) {
    gvt::core::Vector<float> eye = cmd.getValue<float>("eye");
    camNode["eyePoint"] = glm::vec3(eye[0], eye[1], eye[2]);
  }

  if (cmd.isSet("look")) {
    gvt::core::Vector<float> eye = cmd.getValue<float>("look");
    camNode["focus"] = glm::vec3(eye[0], eye[1], eye[2]);
  }
  if (cmd.isSet("wsize")) {
    gvt::core::Vector<int> wsize = cmd.getValue<int>("wsize");
    filmNode["width"] = wsize[0];
    filmNode["height"] = wsize[1];
  }

  gvt::core::DBNodeH schedNode = cntxt->createNodeFromType("Schedule", "Plysched", root.UUID());
  if (cmd.isSet("domain"))
    schedNode["type"] = gvt::render::scheduler::Domain;
  else
    schedNode["type"] = gvt::render::scheduler::Image;

  string adapter("embree");

  if (cmd.isSet("manta")) {
    adapter = "manta";
  } else if (cmd.isSet("optix")) {
    adapter = "optix";
  }

  // adapter
  if (adapter.compare("embree") == 0) {
    std::cout << "Using embree adapter " << std::endl;
#ifdef GVT_RENDER_ADAPTER_EMBREE
    schedNode["adapter"] = gvt::render::adapter::Embree;
#else
    std::cout << "Embree adapter missing. recompile" << std::endl;
    exit(1);
#endif
  } else if (adapter.compare("manta") == 0) {
    std::cout << "Using manta adapter " << std::endl;
#ifdef GVT_RENDER_ADAPTER_MANTA
    schedNode["adapter"] = gvt::render::adapter::Manta;
#else
    std::cout << "Manta adapter missing. recompile" << std::endl;
    exit(1);
#endif
  } else if (adapter.compare("optix") == 0) {
    std::cout << "Using optix adapter " << std::endl;
#ifdef GVT_RENDER_ADAPTER_OPTIX
    schedNode["adapter"] = gvt::render::adapter::Optix;
#else
    std::cout << "Optix adapter missing. recompile" << std::endl;
    exit(1);
#endif
  } else {
    std::cout << "unknown adapter, " << adapter << ", specified." << std::endl;
    exit(1);
  }

  // end db setup

  // use db to create structs needed by system

  // setup gvtCamera from database entries
  gvtPerspectiveCamera mycamera;
  glm::vec3 cameraposition = camNode["eyePoint"].value().tovec3();
  glm::vec3 focus = camNode["focus"].value().tovec3();
  float fov = camNode["fov"].value().toFloat();
  glm::vec3 up = camNode["upVector"].value().tovec3();
  int rayMaxDepth = camNode["rayMaxDepth"].value().toInteger();
  int raySamples = camNode["raySamples"].value().toInteger();
  mycamera.lookAt(cameraposition, focus, up);
  mycamera.setMaxDepth(rayMaxDepth);
  mycamera.setSamples(raySamples);
  mycamera.setFOV(fov);
  mycamera.setFilmsize(filmNode["width"].value().toInteger(), filmNode["height"].value().toInteger());

  t_skip.stop();

  // setup image from database sizes
  Image myimage(mycamera.getFilmSizeWidth(), mycamera.getFilmSizeHeight(), "output");

  mycamera.AllocateCameraRays();
  mycamera.generateRays();

  int schedType = root["Schedule"]["type"].value().toInteger();
  switch (schedType) {
  case gvt::render::scheduler::Image: {
    std::cout << "starting image scheduler" << std::endl;
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
    std::cout << "starting domain scheduler" << std::endl;

    // gvt::render::algorithm::Tracer<DomainScheduler>(mycamera.rays, myimage)();
    std::cout << "starting image scheduler" << std::endl;
    gvt::render::algorithm::Tracer<DomainScheduler> tracer(mycamera.rays, myimage);
    for (int z = 0; z < 100; z++) {
      mycamera.AllocateCameraRays();
      mycamera.generateRays();
      myimage.clear();
      tracer();
    }
    break;
  }
  default: {
    std::cout << "unknown schedule type provided: " << schedType << std::endl;
    break;
  }
  }

  myimage.Write();
  }
  // std::cout << "Observed threads: " << tracker.get_concurrency() << std::endl;

  MPI_Finalize();
}
