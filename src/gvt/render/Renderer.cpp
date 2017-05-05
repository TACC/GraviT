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

gvtRenderer::~gvtRenderer() { delete camera; delete myimage; delete tracer; delete ctx; }
gvtRenderer::gvtRenderer() {
  // First grab the context and sync it.
  //
  ctx = RenderContext::instance(); 
  ctx->syncContext();
  rootnode = ctx->getRootNode();
  datanode = rootnode["Data"];
  instancesnode = rootnode["Instances"];
  lightsnode = rootnode["Lights"];
  cameranode = rootnode["Camera"];
  filmnode = rootnode["Film"];
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
      tracer = new algorithm::Tracer<schedule::DomainScheduler>(camera->rays,*myimage);
      break;
    }
    default: {
      std::cout <<"unknown scuedule type provided: " << rootnode["Schedule"]["type"].value().toInteger();
    }
  }

  
}
void gvtRenderer::render() {
  camera->AllocateCameraRays();
  camera->generateRays();
  (*tracer)();
}

gvtRenderer *gvtRenderer::instance() {
    if(__singleton == nullptr) {
        __singleton = new gvtRenderer();
    }
    return __singleton;
}

