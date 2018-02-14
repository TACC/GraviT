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
#include <gvt/render/Schedulers.h>
#include <gvt/render/Types.h>
#include <gvt/render/api/api.h>
#include <set>
#include <vector>

#include <tbb/task_scheduler_init.h>
#include <thread>

//#ifdef GVT_RENDER_ADAPTER_EMBREE
//#include <gvt/render/adapter/embree/EmbreeMeshAdapter.h>
//#endif
//
//#ifdef GVT_RENDER_ADAPTER_EMBREE_STREAM
//#include <gvt/render/adapter/embree/EmbreeStreamMeshAdapter.h>
//#endif
//
//#ifdef GVT_RENDER_ADAPTER_MANTA
//#include <gvt/render/adapter/manta/MantaMeshAdapter.h>
//#endif
//
//#ifdef GVT_RENDER_ADAPTER_OPTIX
//#include <gvt/render/adapter/optix/OptixMeshAdapter.h>
//#endif
//
#include <gvt/render/algorithm/Tracers.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/data/scene/gvtCamera.h>

#include <gvt/core/comm/communicator/scomm.h>
//
//#include <iostream>
//
//#ifdef __USE_TAU
//#include <TAU.h>
//#endif

#include "ParseCommandLine.h"

using namespace std;
using namespace gvt::render;

using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;
using namespace gvt::render::data::primitives;

void test_bvh(gvtPerspectiveCamera &camera);

int main(int argc, char **argv) {

  // gvtInit(argc, argv);
  api::gvtInit(argc, argv);
  cntx::rcontext &db = cntx::rcontext::instance();

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

  cmd.addoption("embree", ParseCommandLine::NONE, "Embree Adapter Type", 0);
  cmd.addoption("manta", ParseCommandLine::NONE, "Manta Adapter Type", 0);
  cmd.addoption("optix", ParseCommandLine::NONE, "Optix Adapter Type", 0);

  cmd.addconflict("embree", "manta");
  cmd.addconflict("embree", "optix");
  cmd.addconflict("manta", "optix");

  cmd.parse(argc, argv);

  tbb::task_scheduler_init *init;
  if (!cmd.isSet("threads")) {
    init = new tbb::task_scheduler_init(std::thread::hardware_concurrency());
    db.getUnique("threads") = std::thread::hardware_concurrency();
  } else {
    init = new tbb::task_scheduler_init(cmd.get<int>("threads"));
    db.getUnique("threads") = unsigned(cmd.get<int>("threads"));
  }

  if (db.cntx_comm.rank % 2 == 0) {
    std::vector<float> vertex = { 0.5,     0.0,  0.0,  -0.5, 0.5,  0.0,   -0.5,      0.25, 0.433013, -0.5,     -0.25,
                                  0.43013, -0.5, -0.5, 0.0,  -0.5, -0.25, -0.433013, -0.5, 0.25,     -0.433013 };

    std::vector<unsigned> faces = { 1, 2, 3, 1, 3, 4, 1, 4, 5, 1, 5, 6, 1, 6, 7, 1, 7, 2 };
    float kd[] = { 1.f, 1.f, 1.f };

    api::createMesh("conemesh");
    api::addMeshVertices("conemesh", vertex.size() / 3, &vertex[0]);
    api::addMeshTriangles("conemesh", faces.size() / 3, &faces[0]);
    api::addMeshMaterial("conemesh", (unsigned)LAMBERT, kd, 1.f);
    api::finishMesh("conemesh");
  }
  if (db.cntx_comm.rank % 2 == 1 || db.cntx_comm.size == 1) {
    std::vector<float> vertex = { -0.5, -0.5, 0.5,  0.5,  -0.5, 0.5,  0.5,  0.5,  0.5,  -0.5, 0.5,  0.5,

                                  -0.5, -0.5, -0.5, 0.5,  -0.5, -0.5, 0.5,  0.5,  -0.5, -0.5, 0.5,  -0.5,

                                  0.5,  0.5,  0.5,  -0.5, 0.5,  0.5,  0.5,  0.5,  -0.5, -0.5, 0.5,  -0.5,

                                  -0.5, -0.5, 0.5,  0.5,  -0.5, 0.5,  -0.5, -0.5, -0.5, 0.5,  -0.5, -0.5,

                                  0.5,  -0.5, 0.5,  0.5,  0.5,  0.5,  0.5,  -0.5, -0.5, 0.5,  0.5,  -0.5,

                                  -0.5, -0.5, 0.5,  -0.5, 0.5,  0.5,  -0.5, -0.5, -0.5, -0.5, 0.5,  -0.5

    };

    std::vector<unsigned> faces = {
      1,  2,  3,  1,  3,  4,  17, 19, 20, 17, 20, 18, 6,  5,  8,  6,  8,  7,
      23, 21, 22, 23, 22, 24, 10, 9,  11, 10, 11, 12, 13, 15, 16, 13, 16, 14,

    };
    float kd[] = { 1.f, 1.f, 1.f };

    api::createMesh("cubemesh");
    api::addMeshVertices("cubemesh", vertex.size() / 3, &vertex[0]);
    api::addMeshTriangles("cubemesh", faces.size() / 3, &faces[0]);
    api::addMeshMaterial("cubemesh", (unsigned)LAMBERT, kd, 1.f);
    api::finishMesh("cubemesh");
  }

  //

  db.sync();
  // db.printtreebyrank(std::cout);

  if (db.cntx_comm.rank == 0) {
    // create a NxM grid of alternating cones / cubes, offset using i and j
    int instId = 0;
    int ii[2] = { -2, 3 }; // i range
    int jj[2] = { -2, 3 }; // j range
    for (int i = ii[0]; i < ii[1]; i++) {
      for (int j = jj[0]; j < jj[1]; j++) {
        auto m = new glm::mat4(1.f);
        *m = glm::translate(*m, glm::vec3(0.0, i * 0.5, j * 0.5));
        *m = glm::scale(*m, glm::vec3(0.4, 0.4, 0.4));
        string instanceMeshname = (instId % 2) ? "cubemesh" : "conemesh";
        string instanceName = "inst" + std::to_string(instId);

        auto &mi = (*m);

        float mf[] = { mi[0][0], mi[0][1], mi[0][2], mi[0][3], mi[1][0], mi[1][1], mi[1][2], mi[1][3],
                       mi[2][0], mi[2][1], mi[2][2], mi[2][3], mi[3][0], mi[3][1], mi[3][2], mi[3][3] };

        api::addInstance(instanceName, instanceMeshname, mf);
        instId++;
      }
    }
  }

  db.sync();

  auto lpos = glm::vec3(1.0, 0.0, -1.0);
  auto lcolor = glm::vec3(1.0, 1.0, 1.0);

  string lightname = "conelight";

  if (cmd.isSet("lpos")) {
    gvt::core::Vector<float> pos = cmd.getValue<float>("lpos");
    lpos = glm::vec3(pos[0], pos[1], pos[2]);
  }
  if (cmd.isSet("lcolor")) {
    gvt::core::Vector<float> color = cmd.getValue<float>("lcolor");
    lcolor = glm::vec3(color[0], color[1], color[2]);
  }

  api::addPointLight(lightname, glm::value_ptr(lpos), glm::value_ptr(lcolor));
  db.sync();

  // camera bits..
  auto eye = glm::vec3(4.0, 0.0, 0.0);
  if (cmd.isSet("eye")) {
    gvt::core::Vector<float> cameye = cmd.getValue<float>("eye");
    eye = glm::vec3(cameye[0], cameye[1], cameye[2]);
  }
  auto focus = glm::vec3(0.0, 0.0, 0.0);
  if (cmd.isSet("look")) {
    gvt::core::Vector<float> foc = cmd.getValue<float>("look");
    focus = glm::vec3(foc[0], foc[1], foc[2]);
  }
  auto upVector = glm::vec3(0.0, 1.0, 0.0);
  float fov = (float)(45.0 * M_PI / 180.0);

  int rayMaxDepth = (int)1;
  int raySamples = (int)1;
  float jitterWindowSize = (float)0.5;
  string camname = "conecam";
  api::addCamera(camname, glm::value_ptr(eye), glm::value_ptr(focus), glm::value_ptr(upVector), fov, rayMaxDepth,
                  raySamples, jitterWindowSize);

  db.sync();
  // film bits..
  string filmname = "conefilm";
  int width = (int)512;
  int height = (int)512;
  if (cmd.isSet("wsize")) {
    gvt::core::Vector<int> wsize = cmd.getValue<int>("wsize");
    width = wsize[0];
    height = wsize[1];
  }
  string outputpath = "simple";
  if (cmd.isSet("output")) {
    gvt::core::Vector<std::string> output = cmd.getValue<std::string>("output");
    outputpath = output[0];
  }
  api::addFilm(filmname, width, height, outputpath);

  //  db.printtreebyrank(std::cout);
  db.sync();

  // render bits (schedule and adapter)
  string rendername("Enzoschedule");
  int schedtype;
  int adaptertype;
  if (cmd.isSet("domain"))
    schedtype = gvt::render::scheduler::AsyncDomain;
  else
    schedtype = gvt::render::scheduler::AsyncImage;

  string adapter("embree");
  if (cmd.isSet("manta")) {
    adapter = "manta";
  } else if (cmd.isSet("optix")) {
    adapter = "optix";
  }
  if (adapter.compare("embree") == 0) {

#ifdef GVT_RENDER_ADAPTER_EMBREE
    adaptertype = gvt::render::adapter::Embree;
#else
    std::cerr << "Embree adapter missing. recompile" << std::endl;
    exit(1);
#endif
  } else if (adapter.compare("manta") == 0) {

#ifdef GVT_RENDER_ADAPTER_MANTA
    adaptertype = gvt::render::adapter::Manta;
#else
    std::cerr << "Manta adapter missing. recompile" << std::endl;
    exit(1);
#endif
  } else if (adapter.compare("optix") == 0) {
    std::cerr << " optix adapter " << std::endl;
#ifdef GVT_RENDER_ADAPTER_OPTIX
    adaptertype = gvt::render::adapter::Optix;
#else
    std::cerr << "Optix adapter missing. recompile" << std::endl;
    exit(1);
#endif
  } else {
    std::cerr << "unknown adapter, " << adapter << ", specified." << std::endl;
    exit(1);
  }

  api::addRenderer(rendername, adaptertype, schedtype, camname, filmname);
  db.sync();


  std::cout << "All synced" << std::endl;

  api::render(rendername);
  api::writeimage(rendername,"simple");

  //std::shared_ptr<gvt::render::RayTracer> rt;

  //  std::shared_ptr<gvt::render::RayTracer> rt;
  //  int schedType = root["Schedule"]["type"].value().toInteger();
  //  switch (schedType) {
  //  case gvt::render::scheduler::Image: {
  //    rt = std::make_shared<gvt::render::ImageTracer>();
  //    break;
  //  }
  //  case gvt::render::scheduler::Domain: {
  //    rt = std::make_shared<gvt::render::DomainTracer>();
  //    break;
  //  }
  //  default: {
  //    std::cout << "unknown schedule type provided: " << schedType << std::endl;
  //    std::exit(0);
  //    break;
  //  }
  //  }
  //
  //  cntxt->settracer(rt);
  //
  //  std::cout << "Calling tracer" << std::endl;
  //  for (int i = 0; i < 100; i++) {
  //    (*rt)();
  //  }
  //
  //  if (gvt::comm::communicator::instance().id() == 0)
  //    (*rt).getComposite()->write(filmNode["outputPath"].value().toString());
  //  gvt::comm::communicator::instance().terminate();
  //  }
  MPI_Finalize();
}
