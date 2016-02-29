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
#include <algorithm>
#include <gvt/core/Math.h>
#include <gvt/core/mpi/Wrapper.h>
#include <gvt/render/RenderContext.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/Types.h>
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

#include <gvt/render/algorithm/Tracers.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/domain/reader/ObjReader.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/data/scene/gvtCamera.h>

#include <boost/range/algorithm.hpp>

#include <iostream>

using namespace std;
using namespace gvt::render;

using namespace gvt::core::mpi;
using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;
using namespace gvt::render::data::primitives;

int main(int argc, char **argv) {
  tbb::task_scheduler_init init(std::thread::hardware_concurrency());

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

  // add the data - mesh in this case
  gvt::core::DBNodeH dataNodes = cntxt->createNodeFromType("Data", "Data", root.UUID());

  gvt::core::DBNodeH bunnyMeshNode = cntxt->createNodeFromType("Mesh", "bunny", dataNodes.UUID());
  {
    // path assumes binary is run as bin/gvtFileApp
    gvt::render::data::domain::reader::ObjReader objReader("../data/geom/bunny.obj");
    // right now mesh must be converted to gvt format
    Mesh *mesh = objReader.getMesh();
    mesh->generateNormals();

    mesh->computeBoundingBox();
    Box3D *meshbbox = mesh->getBoundingBox();

    // add bunny mesh to the database
    bunnyMeshNode["file"] = string("../data/geom/bunny.obj");
    bunnyMeshNode["bbox"] = (unsigned long long)meshbbox;
    bunnyMeshNode["ptr"] = (unsigned long long)mesh;
  }

  // create the instance
  gvt::core::DBNodeH instNodes = cntxt->createNodeFromType("Instances", "Instances", root.UUID());

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
  auto il = glm::vec3((*m) * glm::vec4(mbox->bounds[0], 1.f));
  auto ih = glm::vec3((*m) * glm::vec4(mbox->bounds[1], 1.f));
  Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
  instnode["bbox"] = (unsigned long long)ibox;
  instnode["centroid"] = ibox->centroid();

  // add a light
  gvt::core::DBNodeH lightNodes = cntxt->createNodeFromType("Lights", "Lights", root.UUID());

  //area Light
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

  // set image width/height
  gvt::core::DBNodeH filmNode = cntxt->createNodeFromType("Film", "film", root.UUID());
  // filmNode["width"] = 4096;
  // filmNode["height"] = 2304;

  // filmNode["width"] = 1920;
  // filmNode["height"] = 1080;

  filmNode["width"] = 512;
  filmNode["height"] = 512;

  // TODO: schedule db design could be modified a bit
  gvt::core::DBNodeH schedNode = cntxt->createNodeFromType("Schedule", "sched", root.UUID());
  schedNode["type"] = gvt::render::scheduler::Image;
// schedNode["type"] = gvt::render::scheduler::Domain;

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
  //
  // start gvt
  //

  // TODO: wrap the following in a static function inside gvt
  // the following starts the system

  // setup gvtCamera from database entries

  int samplesSqureRoot = 1;

  gvtPerspectiveCamera mycamera;
  glm::vec3 cameraposition = camNode["eyePoint"].value().tovec3();
  glm::vec3 focus = camNode["focus"].value().tovec3();
  float fov = camNode["fov"].value().toFloat();
  glm::vec3 up = camNode["upVector"].value().tovec3();
  mycamera.setSamples(samplesSqureRoot);
  mycamera.setJitterWindowSize(0.00);
  mycamera.lookAt(cameraposition, focus, up);
  mycamera.setFOV(fov);
  mycamera.setFilmsize(filmNode["width"].value().toInteger(), filmNode["height"].value().toInteger());

  // setup image from database sizes
  Image myimage(mycamera.getFilmSizeWidth(), mycamera.getFilmSizeHeight(), "bunny");

  mycamera.AllocateCameraRays();
  mycamera.generateRays();

  int schedType = root["Schedule"]["type"].value().toInteger();
  switch (schedType) {
  case gvt::render::scheduler::Image: {
    std::cout << "starting image scheduler" << std::endl;
    auto tracer = gvt::render::algorithm::Tracer<ImageScheduler>(mycamera.rays, myimage);
    tracer.sample_ratio = 1.0/float(samplesSqureRoot*samplesSqureRoot);
    tracer();
    break;
  }
  case gvt::render::scheduler::Domain: {
    std::cout << "starting domain scheduler" << std::endl;
    auto tracer = gvt::render::algorithm::Tracer<DomainScheduler>(mycamera.rays, myimage);
    tracer.sample_ratio = 1.0/float(samplesSqureRoot*samplesSqureRoot);
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
