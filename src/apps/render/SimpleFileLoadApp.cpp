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
//
// Simple gravit application.
// Load some geometry and render it.
//
#if 0 // TODO: pnav - update SimpleFileLoadApp to use new context
#include <algorithm>
#include <gvt/core/Math.h>
#include <gvt/render/RenderContext.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/Types.h>
#include <set>
#include <vector>

#include <tbb/task_scheduler_init.h>
#include <thread>

#ifdef GVT_RENDER_ADAPTER_EMBREE
#include <gvt/render/adapter/embree/EmbreeMeshAdapter.h>
#endif

#ifdef GVT_RENDER_ADAPTER_EMBREE_STREAM
#include <gvt/render/adapter/embree/EmbreeStreamMeshAdapter.h>
#endif

#ifdef GVT_RENDER_ADAPTER_MANTA
#include <gvt/render/adapter/manta/MantaMeshAdapter.h>
#endif

#ifdef GVT_RENDER_ADAPTER_OPTIX
#include <gvt/render/adapter/optix/OptixMeshAdapter.h>
#endif

#include <gvt/render/algorithm/Tracers.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/reader/ObjReader.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/data/scene/gvtCamera.h>


#include <iostream>

#include "ParseCommandLine.h"

using namespace std;
using namespace gvt::render;

using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;
using namespace gvt::render::data::primitives;

int main(int argc, char **argv) {
  ParseCommandLine cmd("gvtFileLoad");
  cmd.addoption("obj", ParseCommandLine::PATH, "Location of Obj object", 1);
  cmd.addoption("wsize", ParseCommandLine::INT, "Window size", 2);
  cmd.addoption("eye", ParseCommandLine::FLOAT, "Camera position", 3);
  cmd.addoption("look", ParseCommandLine::FLOAT, "Camera look at", 3);
  cmd.addoption("up", ParseCommandLine::FLOAT, "Camera up vector", 3);
  cmd.addoption("point-light", ParseCommandLine::FLOAT, "Point light position and color (px,py,pz,cx,cy,cz)", 6);
  cmd.addoption("image", ParseCommandLine::NONE, "Use embeded scene", 0);
  cmd.addoption("domain", ParseCommandLine::NONE, "Use embeded scene", 0);
  cmd.addoption("threads", ParseCommandLine::INT, "Number of threads to use (default number cores + ht)", 1);
  cmd.addoption("output", ParseCommandLine::PATH, "Output Image Path", 1);

  cmd.addoption("embree", ParseCommandLine::NONE, "Embree Adapter Type", 0);
  cmd.addoption("embree-stream", ParseCommandLine::NONE, "Embree Adapter Type (Stream)", 0);
  cmd.addoption("manta", ParseCommandLine::NONE, "Manta Adapter Type", 0);
  cmd.addoption("optix", ParseCommandLine::NONE, "Optix Adapter Type", 0);

  cmd.addoption("material", ParseCommandLine::INT,
                "Material type (0: Lambert, 1: Phong, 2: Blinn, 3: Embree_Metal, 4: Embree_Velvet, 5: Embree_Matte )",
                1);

  cmd.addconflict("embree", "manta");
  cmd.addconflict("embree", "optix");
  cmd.addconflict("embree-stream", "manta");
  cmd.addconflict("embree-stream", "optix");
  cmd.addconflict("manta", "optix");

  cmd.addconflict("image", "domain");
  cmd.parse(argc, argv);

  tbb::task_scheduler_init *init;
  if (!cmd.isSet("threads")) {
    init = new tbb::task_scheduler_init(std::thread::hardware_concurrency());
  } else {
    init = new tbb::task_scheduler_init(cmd.get<int>("threads"));
  }

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

  if (rank == 0) {
    gvt::core::DBNodeH dataNodes = cntxt->addToSync(cntxt->createNodeFromType("Data", "Data", root.UUID()));
    cntxt->addToSync(cntxt->createNodeFromType("Mesh", "bunny", dataNodes.UUID()));
    cntxt->addToSync(cntxt->createNodeFromType("Instances", "Instances", root.UUID()));
  }

  cntxt->syncContext();

  gvt::core::DBNodeH dataNodes = root["Data"];
  gvt::core::DBNodeH instNodes = root["Instances"];

  gvt::core::DBNodeH bunnyMeshNode = dataNodes.getChildren()[0];

  {
    // path assumes binary is run as bin/gvtFileApp
    std::string objPath = std::string("../data/geom/bunny.obj");
    if (cmd.isSet("obj")) {
      objPath = cmd.getValue<std::string>("obj")[0];
    }

    int material_type = 0;
    if (cmd.isSet("material")) {
      gvt::core::Vector<int> type = cmd.getValue<int>("material");
      material_type = type[0];
    }

    gvt::render::data::domain::reader::ObjReader objReader(objPath, material_type);

    // right now mesh must be converted to gvt format
    Mesh *mesh = objReader.getMesh();
    mesh->generateNormals();

    mesh->computeBoundingBox();
    Box3D *meshbbox = mesh->getBoundingBox();

    // add bunny mesh to the database

    bunnyMeshNode["file"] = objPath;
    bunnyMeshNode["bbox"] = (unsigned long long)meshbbox;
    bunnyMeshNode["ptr"] = (unsigned long long)mesh;

    gvt::core::DBNodeH loc = cntxt->createNode("rank", rank);
    bunnyMeshNode["Locations"] += loc;

    cntxt->addToSync(bunnyMeshNode);
  }

  cntxt->syncContext();

  // create the instance
  if (rank == 0) {
    gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance", "inst", instNodes.UUID());
    gvt::core::DBNodeH meshNode = bunnyMeshNode;
    Box3D *mbox = (Box3D *)meshNode["bbox"].value().toULongLong();

    instnode["id"] = 0; // unique id per instance
    instnode["meshRef"] = meshNode.UUID();

    // transform bunny
    float scale = 1.0;
    auto m = new glm::mat4(1.f);
    auto minv = new glm::mat4(1.f);
    auto normi = new glm::mat3(1.f);
    //*m = glm::translate(*m, glm::vec3(0, 0, 0));
    //*m *glm::mat4::createTranslation(0.0, 0.0, 0.0);
    //*m = *m * glm::mat4::createScale(scale, scale, scale);
    *m = glm::scale(*m, glm::vec3(scale, scale, scale));

    instnode["mat"] = (unsigned long long)m;
    *minv = glm::inverse(*m);
    instnode["matInv"] = (unsigned long long)minv;
    *normi = glm::transpose(glm::inverse(glm::mat3(*m)));
    instnode["normi"] = (unsigned long long)normi;

    // transform mesh bounding box
    auto il = glm::vec3((*m) * glm::vec4(mbox->bounds_min, 1.f));
    auto ih = glm::vec3((*m) * glm::vec4(mbox->bounds_max, 1.f));
    Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
    instnode["bbox"] = (unsigned long long)ibox;
    instnode["centroid"] = ibox->centroid();
    cntxt->addToSync(instnode);
  }

  cntxt->syncContext();

  // add a light
  gvt::core::DBNodeH lightNodes = cntxt->createNodeFromType("Lights", "Lights", root.UUID());

  // area Light
  // gvt::core::DBNodeH lightNode = cntxt->createNodeFromType("AreaLight", "light", lightNodes.UUID());
  // lightNode["position"] = glm::vec3(-0.2, 0.1, 0.9, 0.0);
  // lightNode["color"] = glm::vec3(1.0, 1.0, 1.0, 0.0);
  // lightNode["normal"] = glm::vec3(0.0, 0.0, 1.0, 0.0);
  // lightNode["width"] = float(0.05);
  // lightNode["height"] = float(0.05);

  gvt::core::DBNodeH lightNode = cntxt->createNodeFromType("PointLight", "light", lightNodes.UUID());
  lightNode["position"] = glm::vec3(0.0, 0.1, 0.5);
  lightNode["color"] = glm::vec3(1.0, 1.0, 1.0);

  // set the camera
  gvt::core::DBNodeH camNode = cntxt->createNodeFromType("Camera", "cam", root.UUID());
  camNode["eyePoint"] = glm::vec3(0.0, 0.1, 0.3);
  camNode["focus"] = glm::vec3(0.0, 0.1, -0.3);
  camNode["upVector"] = glm::vec3(0.0, 1.0, 0.0);
  camNode["fov"] = (float)(45.0 * M_PI / 180.0);
  camNode["rayMaxDepth"] = (int)1;
  camNode["raySamples"] = (int)1;
  camNode["jitterWindowSize"] = (float)0;

  // set image width/height
  gvt::core::DBNodeH filmNode = cntxt->createNodeFromType("Film", "film", root.UUID());
  filmNode["width"] = 512;
  filmNode["height"] = 512;
  filmNode["outputPath"] = cmd.isSet("obj") ? std::string("output") : std::string("bunny");

  if (cmd.isSet("eye")) {
    gvt::core::Vector<float> eye = cmd.getValue<float>("eye");
    camNode["eyePoint"] = glm::vec3(eye[0], eye[1], eye[2]);
  }

  if (cmd.isSet("look")) {
    gvt::core::Vector<float> eye = cmd.getValue<float>("look");
    camNode["focus"] = glm::vec3(eye[0], eye[1], eye[2]);
  }

  if (cmd.isSet("up")) {
    gvt::core::Vector<float> up = cmd.getValue<float>("up");
    camNode["upVector"] = glm::vec3(up[0], up[1], up[2]);
  }

  if (cmd.isSet("point-light")) {
    gvt::core::Vector<float> light = cmd.getValue<float>("point-light");
    lightNode["position"] = glm::vec3(light[0], light[1], light[2]);
    lightNode["color"] = glm::vec3(light[3], light[4], light[5]);
  }

  if (cmd.isSet("wsize")) {
    gvt::core::Vector<int> wsize = cmd.getValue<int>("wsize");
    filmNode["width"] = wsize[0];
    filmNode["height"] = wsize[1];
  }
  if (cmd.isSet("output")) {
    gvt::core::Vector<std::string> output = cmd.getValue<std::string>("output");
    filmNode["outputPath"] = output[0];
  }

  gvt::core::DBNodeH schedNode = cntxt->createNodeFromType("Schedule", "Enzosched", root.UUID());
  if (cmd.isSet("domain"))
    schedNode["type"] = gvt::render::scheduler::Domain;
  else
    schedNode["type"] = gvt::render::scheduler::Image;

// schedNode["type"] = gvt::render::scheduler::Domain;

#ifdef GVT_RENDER_ADAPTER_EMBREE
  int adapterType = gvt::render::adapter::Embree;
#elif GVT_RENDER_ADAPTER_EMBREE_STREAM
  int adapterType = gvt::render::adapter::EmbreeStream;
#elif GVT_RENDER_ADAPTER_MANTA
  int adapterType = gvt::render::adapter::Manta;
#elif GVT_RENDER_ADAPTER_OPTIX
  int adapterType = gvt::render::adapter::Optix;
#else
  GVT_ERR_MESSAGE("ERROR: missing valid adapter");
#endif

  schedNode["adapter"] = adapterType;
  //
  // start gvt
  //

  // TODO: wrap the following in a static function inside gvt
  // the following starts the system

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

  // setup image from database sizes
  Image myimage(mycamera.getFilmSizeWidth(), mycamera.getFilmSizeHeight(), filmNode["outputPath"].value().toString());

  mycamera.AllocateCameraRays();
  mycamera.generateRays();

  int schedType = root["Schedule"]["type"].value().toInteger();
  switch (schedType) {
  case gvt::render::scheduler::Image: {
    std::cout << "starting image scheduler" << std::endl;
    gvt::render::algorithm::Tracer<ImageScheduler> tracer(mycamera.rays, myimage);
    tracer.sample_ratio = 1.0 / float(raySamples * raySamples);
    tracer();
    break;
  }
  case gvt::render::scheduler::Domain: {
    std::cout << "starting domain scheduler" << std::endl;
    gvt::render::algorithm::Tracer<DomainScheduler> tracer(mycamera.rays, myimage);
    tracer.sample_ratio = 1.0 / float(raySamples * raySamples);
    tracer();
    break;
  }
  default: {
    std::cout << "unknown schedule type provided: " << schedType << std::endl;
    break;
  }
  }

  myimage.Write();
  if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();
}
#endif // TODO: pnav - update SimpleFileLoadApp to use new context