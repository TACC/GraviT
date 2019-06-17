/* =======================================================================================
 *    This file is released as part of GraviT - scalable, platform independent ray tracing
 *    tacc.github.io/GraviT
 *
 *    Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas at Austin
 *    All rights reserved.
 *
 *    Licensed under the BSD 3-Clause License, (the "License"); you may not use this file
 *    except in compliance with the License.
 *    A copy of the License is included with this software in the file LICENSE.
 *    If your copy does not contain the License, you may obtain a copy of the License at:
 *
 *    http://opensource.org/licenses/BSD-3-Clause
 *
 *    Unless required by applicable law or agreed to in writing, software distributed under
 *    the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *    KIND, either express or implied.
 *    See the License for the specific language governing permissions and limitations under
 *    limitations under the License.
 *
 *    GraviT is funded in part by the US National Science Foundation under awards ACI-1339863,
 *    ACI-1339881 and ACI-1339840
 *    ======================================================================================= */
#include <gvt/render/Renderer.h>

using namespace gvt::render;

gvtRenderer *gvtRenderer::__singleton = nullptr;

gvtRenderer::~gvtRenderer() {}
gvtRenderer::gvtRenderer() {
  // First grab the context and sync it.
  //
  //  ctx = RenderContext::instance();
  //  ctx->syncContext();
  //  rootnode = ctx->getRootNode();
  //  //ctx->database()->printTree(rootnode.UUID(), 10, std::cout);
  //  datanode = rootnode["Data"];
  //  instancesnode = rootnode["Instances"];
  //  lightsnode = rootnode["Lights"];
  //  cameranode = rootnode["Camera"];
  //  filmnode = rootnode["Film"];

  cntx::rcontext &db = cntx::rcontext::instance();

#if 0
  auto& camera;


  // build out concrete instances of scene objects. 
  // camera 
  camera = new data::scene::gvtPerspectiveCamera();
  glm::vec3 cameraposition = cameranode["eyePoint"].value().tovec3();
  glm::vec3 focus = cameranode["focus"].value().tovec3();
  glm::vec3 up = cameranode["upVector"].value().tovec3();
  //
  camera->setMaxDepth(cameranode["rayMaxDepth"].value().toInteger());
  camera->setSamples(cameranode["raySamples"].value().toInteger());
  camera->setJitterWindowSize(cameranode["jitterWindowSize"].value().toFloat());
  camera->lookAt(cameraposition, focus, up);
  camera->setFOV(cameranode["fov"].value().toFloat());
  camera->setFilmsize(filmnode["width"].value().toInteger(),filmnode["height"].value().toInteger());
  // image plane setup. 
  myimage = new data::scene::Image(camera->getFilmSizeWidth(),camera->getFilmSizeHeight(),filmnode["outputPath"].value().toString());
  std::cout << " film output file " << filmnode["outputPath"].value().toString() << std::endl;
  // allocate rays (needed by tracer constructor)
  camera->AllocateCameraRays();
  camera->generateRays();
  // now comes the tricky part. setting up the renderer itself. 
  switch(rootnode["Schedule"]["type"].value().toInteger()) {
    case scheduler::Image: {
      tracer = new  algorithm::Tracer<schedule::ImageScheduler>(camera->rays,*myimage);
      break;
    }
    case scheduler::Domain: {
      if(rootnode["Schedule"]["adapter"].value().toInteger() == gvt::render::adapter::Ospray) { 
          int numargs = 1;
          char **arguments = NULL;
          gvt::render::adapter::ospray::data::OSPRayAdapter::initospray(&numargs,arguments);
      tracer = new algorithm::Tracer<schedule::DomainScheduler>(camera->rays,*myimage);
      } else {
      tracer = new algorithm::Tracer<schedule::DomainScheduler>(camera->rays,*myimage);
      }
      break;
    }
    default: {
    }
  }
#endif
}

void gvtRenderer::reload(std::string const &name) {

    std::cerr << " reloading gvtRenderer " << name << std::endl;
  if (name == current_scheduler) return;
  cntx::rcontext &db = cntx::rcontext::instance();

  auto &ren = db.getUnique(name);
  GVT_ASSERT(!ren.getid().isInvalid(), "Suplied renderer " << name << " is not valid");

  auto &cam = db.getUnique(db.getChild(ren, "camera"));
  auto &fil = db.getUnique(db.getChild(ren, "film"));

  camera = std::make_shared<data::scene::gvtPerspectiveCamera>();

  glm::vec3 cameraposition = db.getChild(cam, "eyePoint");
  glm::vec3 focus = db.getChild(cam, "focus");
  glm::vec3 up = db.getChild(cam, "upVector"); // cameranode["upVector"].value().tovec3();

  camera->setMaxDepth(db.getChild(cam, "rayMaxDepth"));
  camera->setSamples(db.getChild(cam, "raySamples"));
  camera->setJitterWindowSize((float)db.getChild(cam, "jitterWindowSize"));
  camera->lookAt(cameraposition, focus, up);
  camera->setFOV((float)db.getChild(cam, "fov"));

  camera->setFilmsize(db.getChild(fil, "width"), db.getChild(fil, "height"));

  // image plane setup.
  myimage = std::make_shared<composite::IceTComposite>(camera->getFilmSizeWidth(), camera->getFilmSizeHeight());
  // allocate rays (needed by tracer constructor)
//  camera->AllocateCameraRays();
//  camera->generateRays();
  // now comes the tricky part. setting up the renderer itself.

  volume = db.getChild(ren, "volume").to<bool>();

  std::cerr << "sched type "<< db.getChild(ren,"type").to<int>()<< std::endl;
  switch (db.getChild(ren, "type").to<int>()) {
  case scheduler::Image: {
    std::cerr << " image shed " << std::endl;
    tracersync = std::make_shared<algorithm::Tracer<schedule::ImageScheduler> >(camera, myimage, cam, fil, name);
    db.tracer = tracerasync = nullptr;
    break;
  }
  case scheduler::Domain: {
    std::cerr << " domain shed " << std::endl;
    tracersync = std::make_shared<algorithm::Tracer<schedule::DomainScheduler> >(camera, myimage, cam, fil, name);
    db.tracer = tracerasync = nullptr;
    break;
  }
  case scheduler::AsyncDomain: {
    std::cerr << " asyncd shed " << std::endl;
    db.tracer = tracerasync = std::make_shared<gvt::render::DomainTracer>(name,camera,myimage);
    tracersync = nullptr;
    break;
  }
  case scheduler::AsyncImage: {
    std::cerr << " asynci shed " << std::endl;
    db.tracer = tracerasync = std::make_shared<gvt::render::ImageTracer>(name,camera,myimage);
    tracersync = nullptr;
    break;
  }
  default: {}
  }

  // db.tracer = tracer;
}

void gvtRenderer::render(std::string const &name) {
  reload(name);
  std::cerr << " gvtRenderer allocate camera rays " << std::endl;
  camera->AllocateCameraRays();
  std::cerr << " gvtRenderer generate camera rays " << std::endl;
  camera->generateRays(volume);
  if (tracersync) {
  std::cerr << " gvtRenderer synchronours tracer call " << std::endl;
    (*tracersync.get())();
  } else if (tracerasync) {
  std::cerr << " gvtRenderer asynchronours tracer call " << std::endl;
    (*tracerasync.get())();
  }
}
void gvtRenderer::WriteImage(std::string const &name) { myimage->write(name); }

gvtRenderer *gvtRenderer::instance() {
  if (__singleton == nullptr) {
      std::cout << "creatin a new gvtRenderer " << std::endl;
    __singleton = new gvtRenderer();
  }
  return __singleton;
}
