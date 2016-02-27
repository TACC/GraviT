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
// GeometryDomain.C
//

#include <gvt/render/data/domain/GeometryDomain.h>

#include <gvt/render/data/domain/reader/PlyReader.h>

#include <boost/timer/timer.hpp>

using namespace gvt::render::data::domain;
using namespace gvt::render::data::primitives;
using namespace gvt::render::data::scene;
using namespace gvt::core::math;

void GeometryDomain::free() {
  if (!isLoaded) return;
  for (int i = lights.size() - 1; i >= 0; i--) {
    delete lights[i];
    lights.pop_back();
  }
  if (mesh) {
    delete mesh;
    mesh = NULL;
  }
  isLoaded = false;
}

bool GeometryDomain::load() {
  if (isLoaded) return true;
  GVT_ASSERT(filename == "", "No filename");
  {
    GVT_DEBUG(DBG_LOW, "GeometryDomain::load() loading ply file");
    boost::timer::auto_cpu_timer t;
    gvt::render::data::domain::reader::PlyReader reader(filename);
    mesh = reader.getMesh();
  }

  lights.push_back(new PointLight(Point4f(5.0, 5.0, 5.0, 1.f), Color(1.f, 1.f, 1.f, 1.f)));
  mesh->setMaterial(new Lambert(Color(1.f, .0f, .0f, 1.f)));
  boundingBox = *(mesh->getBoundingBox());
  isLoaded = true;
  return isLoaded;
}

namespace gvt {
namespace render {
namespace data {
namespace domain {
std::ostream &operator<<(std::ostream &os, GeometryDomain const &d) {
  os << "geometry domain @ addr " << (void *)&d << std::endl;
  os << "    XXX not yet implemented XXX" << std::endl;
  os << std::flush;

  return os;
}
}
}
}
} // namepsace domain} namespace data} namespace render} namespace gvt}
