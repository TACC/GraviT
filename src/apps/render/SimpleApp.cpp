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
#include <gvt/core/context/Variant.h>
#include <gvt/render/RenderContext.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/Types.h>
#include <gvt/render/data/Domains.h>
#include <gvt/render/Renderer.h>
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

#include <boost/range/algorithm.hpp>

#include <iostream>

#ifdef __USE_TAU
#include <TAU.h>
#endif

#include "ParseCommandLine.h"
#define USEAPI
#ifdef USEAPI
#include <gvt/render/api/api.h>
#endif
using namespace std;
using namespace gvt::render;

using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;
using namespace gvt::render::data::primitives;

void test_bvh(gvtPerspectiveCamera &camera);

int main(int argc, char **argv) {

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
  } else {
    init = new tbb::task_scheduler_init(cmd.get<int>("threads"));
  }

  gvtInit(argc,argv); 

// create a cone mesh with a particular material
  {
    Material *m = new Material();
    m->type = LAMBERT;
    // m->type = EMBREE_MATERIAL_MATTE;
    m->kd = glm::vec3(1.0, 1.0, 1.0);
    m->ks = glm::vec3(1.0, 1.0, 1.0);
    m->alpha = 0.5;

    // m->type = EMBREE_MATERIAL_METAL;
    // copper metal
    m->eta = glm::vec3(.19, 1.45, 1.50);
    m->k = glm::vec3(3.06, 2.40, 1.88);
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
  string meshname("conemesh");
  std::cerr << "adding conemesh" << std::endl;
  addMesh(meshbbox,mesh,meshname); 
  }

// and now a cube
  {

    Material *m = new Material();
    m->type = LAMBERT;
    // m->type = EMBREE_MATERIAL_MATTE;
    m->kd = glm::vec3(1.0, 1.0, 1.0);
    m->ks = glm::vec3(1.0, 1.0, 1.0);
    m->alpha = 0.5;

    // m->type = EMBREE_MATERIAL_METAL;
    // copper metal
    m->eta = glm::vec3(.19, 1.45, 1.50);
    m->k = glm::vec3(3.06, 2.40, 1.88);
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

  string meshname("cubemesh");
  std::cerr << "adding cubemesh" << std::endl;
  addMesh(meshbbox,mesh,meshname); 
  }

// this should happen first thing in the render call

  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
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
        string instanceName = "inst" + std::to_string(instId) ;
	addInstance(instanceName,instanceMeshname,instId,m);
        instId++;
      }
    }
  }

  auto lpos =  glm::vec3(1.0,0.0,-1.0);
  auto lcolor =  glm::vec3(1.0,1.0,1.0);
  string lightname = "conelight";
  if (cmd.isSet("lpos")) {
    gvt::core::Vector<float> pos = cmd.getValue<float>("lpos");
    lpos = glm::vec3(pos[0], pos[1], pos[2]);
  }
  if (cmd.isSet("lcolor")) {
    gvt::core::Vector<float> color = cmd.getValue<float>("lcolor");
    lcolor = glm::vec3(color[0], color[1], color[2]);
  }
  std::cerr <<"add point light"<< std::endl;
  addPointLight(lightname,lpos,lcolor);

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
  std::cerr << " add Camera " << std::endl;
  addCamera(camname,eye,focus,upVector,fov,rayMaxDepth,raySamples,jitterWindowSize);
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
  std::cerr << " add film " << std::endl;
  addFilm(filmname,width,height,outputpath);
// render bits (schedule and adapter)
  std::cerr << "render bits" << std::endl;
  string rendername("Enzoschedule");
  int schedtype;
  int adaptertype;
  if (cmd.isSet("domain"))
    schedtype = gvt::render::scheduler::Domain;
  else
    schedtype = gvt::render::scheduler::Image;

  string adapter("embree");
  if (cmd.isSet("manta")) {
    adapter = "manta";
  } else if (cmd.isSet("optix")) {
    adapter = "optix";
  }
  if (adapter.compare("embree") == 0) {
    std::cerr << " embree adapter " << std::endl;
#ifdef GVT_RENDER_ADAPTER_EMBREE
    adaptertype = gvt::render::adapter::Embree;
#else
    std::cerr << "Embree adapter missing. recompile" << std::endl;
    exit(1);
#endif
  } else if (adapter.compare("manta") == 0) {
    std::cerr << " manta adapter " << std::endl;
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
  std::cerr << "simplsApp: database setup complete - adding renderer" << std::endl;
  addRenderer(rendername,adaptertype,schedtype);
  gvt::render::gvtRenderer *ren = gvt::render::gvtRenderer::instance();
  ren->render();
  ren->WriteImage();

  if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();
}
