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
// Dataset.h
//

#ifndef GVT_RENDER_DATA_DATASET_H
#define GVT_RENDER_DATA_DATASET_H

#include <gvt/core/Debug.h>
#include <gvt/render/data/Domains.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/Camera.h>
#include <gvt/render/data/scene/gvtCamera.h>
#include <gvt/render/data/scene/Light.h>
#include <gvt/render/data/accel/AbstractAccel.h>

#include <algorithm>
#include <cfloat>
#include <map>
#include <string>
#include <vector>
#include <map>

namespace gvt {
namespace render {
class Attributes;
namespace data {
/// global data representation
/** container for global data information.
A Dataset contains a set of Domains and the acceleration structure for the domain extents,
as well as global informaiton such as light sources
*/
class Dataset {
public:
  Dataset();
  ~Dataset();

  virtual bool init();
  virtual int size();
  virtual bool intersect(gvt::render::actor::Ray &r, gvt::render::actor::isecDomList &inter);
  virtual gvt::render::data::domain::AbstractDomain *getDomain(size_t id);
  virtual gvt::render::data::scene::Light *getLight(size_t id);
  virtual int addDomain(gvt::render::data::domain::AbstractDomain *dom);
  virtual int addLight(gvt::render::data::scene::Light *ls);
  void makeAccel(gvt::render::Attributes &rta);
  void makeAccel();

  friend std::ostream &operator<<(std::ostream &, Dataset const &);

public:
  gvt::render::data::primitives::Box3D dataSetBB;
  gvt::render::data::scene::Camera camera;
  gvt::render::data::scene::gvtPerspectiveCamera GVTCamera;
  std::vector<gvt::render::data::domain::AbstractDomain *> domainSet;
  std::vector<gvt::render::data::scene::Light *> lightSet;
  gvt::render::data::accel::AbstractAccel *acceleration;
  std::map<std::string, gvt::render::data::primitives::Mesh *> objMeshes;
};
}
}
}

#endif // GVT_RENDER_DATA_DATASET_H
