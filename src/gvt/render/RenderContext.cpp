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
#include "gvt/render/RenderContext.h"

#include "gvt/core/Debug.h"

using gvt::core::DBNodeH;
using gvt::core::String;
using gvt::core::Uuid;

using namespace gvt::render;

RenderContext::RenderContext() : gvt::core::CoreContext() {}

void RenderContext::CreateContext() {
  if (!__singleton) {
    __singleton = new RenderContext();
  }
}

RenderContext *RenderContext::instance() { return static_cast<RenderContext *>(CoreContext::instance()); }

RenderContext::~RenderContext() {}

DBNodeH RenderContext::createNodeFromType(String type, String name, Uuid parent) {

  DBNodeH n = gvt::core::CoreContext::createNode(type, name, parent);

  // TODO - make these for GraviT
  if (type == String("Camera")) {
    n += gvt::core::CoreContext::createNode("focus");
    n += gvt::core::CoreContext::createNode("eyePoint");
    n += gvt::core::CoreContext::createNode("upVector");
    n += gvt::core::CoreContext::createNode("fov");
    n += gvt::core::CoreContext::createNode("cam2wrld");
    n += gvt::core::CoreContext::createNode("rayMaxDepth");
    n += gvt::core::CoreContext::createNode("raySamples");
    n += gvt::core::CoreContext::createNode("jitterWindowSize");
  } else if (type == String("Film")) {
    n += gvt::core::CoreContext::createNode("width");
    n += gvt::core::CoreContext::createNode("height");
    n += gvt::core::CoreContext::createNode("outputPath");
  } else if (type == String("View")) // TODO: remove view db entries
  {
    n += gvt::core::CoreContext::createNode("width");
    n += gvt::core::CoreContext::createNode("height");
    n += gvt::core::CoreContext::createNode("viewAngle");
    n += gvt::core::CoreContext::createNode("camera", glm::vec3());
    n += gvt::core::CoreContext::createNode("focus", glm::vec3());
    n += gvt::core::CoreContext::createNode("up", glm::vec3());
    n += gvt::core::CoreContext::createNode("parallelScale");
    n += gvt::core::CoreContext::createNode("nearPlane");
    n += gvt::core::CoreContext::createNode("farPlane");
  } else if (type == String("Dataset")) // TODO: remove dataset db entries
  {
    GVT_DEBUG(DBG_ALWAYS, "renderctx: db: ERROR: should not be creating a Dataset type");
    n += gvt::core::CoreContext::createNode("schedule");
    n += gvt::core::CoreContext::createNode("render_type");
    n += gvt::core::CoreContext::createNode("topology");
    n += gvt::core::CoreContext::createNode("Dataset_Pointer");
    n += gvt::core::CoreContext::createNode("accel_type");
    n += gvt::core::CoreContext::createNode("Mesh_Pointer");
  } else if (type == String("Attributes")) // TODO: remove attributes db entries
  {
    GVT_DEBUG(DBG_ALWAYS, "renderctx: db: ERROR: should not be creating an Attributes type");
    n["Views"] += gvt::core::CoreContext::createNodeFromType("View");
    n += gvt::core::CoreContext::createNode("renderType", "surface");
    n += gvt::core::CoreContext::createNode("schedule", "Image");
  } else if (type == String("Mesh")) {
    n += gvt::core::CoreContext::createNode("file");
    n += gvt::core::CoreContext::createNode("ptr");
    n += gvt::core::CoreContext::createNode("bbox");
  } else if (type == String("Instance")) {
    n += gvt::core::CoreContext::createNode("id");
    n += gvt::core::CoreContext::createNode("meshRef");
    n += gvt::core::CoreContext::createNode("bbox");
    n += gvt::core::CoreContext::createNode("centroid");
    n += gvt::core::CoreContext::createNode("mat");
    n += gvt::core::CoreContext::createNode("matInv");
    n += gvt::core::CoreContext::createNode("normi");
  } else if (type == String("PointLight")) {
    n += gvt::core::CoreContext::createNode("position");
    n += gvt::core::CoreContext::createNode("color");
  } else if (type == String("AreaLight")) {
    n += gvt::core::CoreContext::createNode("position");
    n += gvt::core::CoreContext::createNode("color");
    n += gvt::core::CoreContext::createNode("normal");
    n += gvt::core::CoreContext::createNode("height");
    n += gvt::core::CoreContext::createNode("width");
  } else if (type == String("Schedule")) {
    n += gvt::core::CoreContext::createNode("type");
    n += gvt::core::CoreContext::createNode("adapter");
  }

  return n;
}
