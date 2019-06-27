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

  cntx::rcontext &db = cntx::rcontext::instance();

}

void gvtRenderer::reload(std::string const &name) {

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

  switch (db.getChild(ren, "type").to<int>()) {
  case scheduler::Image: {
    tracersync = std::make_shared<algorithm::Tracer<schedule::ImageScheduler> >(camera, myimage, cam, fil, name);
    db.tracer = tracerasync = nullptr;
    break;
  }
  case scheduler::Domain: {
    tracersync = std::make_shared<algorithm::Tracer<schedule::DomainScheduler> >(camera, myimage, cam, fil, name);
    db.tracer = tracerasync = nullptr;
    break;
  }
  case scheduler::AsyncDomain: {
    db.tracer = tracerasync = std::make_shared<gvt::render::DomainTracer>(name,camera,myimage);
    tracersync = nullptr;
    break;
  }
  case scheduler::AsyncImage: {
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
  camera->AllocateCameraRays();
  camera->generateRays(volume);
  if (tracersync) {
    (*tracersync.get())();
  } else if (tracerasync) {
    (*tracerasync.get())();
  }
}
void gvtRenderer::WriteImage(std::string const &name) { myimage->write(name); }

gvtRenderer *gvtRenderer::instance() {
  if (__singleton == nullptr) {
    __singleton = new gvtRenderer();
  }
  return __singleton;
}
