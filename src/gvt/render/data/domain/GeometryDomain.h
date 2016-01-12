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
// GeometryDomain.h
//

#ifndef GVT_RENDER_DATA_DOMAIN_GEOMETRY_DOMAIN_H
#define GVT_RENDER_DATA_DOMAIN_GEOMETRY_DOMAIN_H

#include <gvt/render/data/domain/AbstractDomain.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/Light.h>

#include <iostream>
#include <vector>

namespace gvt {
namespace render {
namespace data {
namespace domain {
/// atomic geometry data unit for GraviT internal use
/** Domain for geometry data.
\sa AbstractDomain, VolumeDomain
*/
class GeometryDomain : public AbstractDomain {
public:
  GeometryDomain(std::string filename = "",
                 gvt::core::math::AffineTransformMatrix<float> m = gvt::core::math::AffineTransformMatrix<float>(true))
      : AbstractDomain(m), mesh(NULL), filename(filename) {
    if (filename != "") {
      load();
    }
  }

  GeometryDomain(gvt::render::data::primitives::Mesh *mesh,
                 gvt::core::math::AffineTransformMatrix<float> m = gvt::core::math::AffineTransformMatrix<float>(true))
      : AbstractDomain(m), mesh(NULL), filename("") {
    if (mesh != NULL) {
      this->mesh = mesh;
      this->boundingBox = *(mesh->getBoundingBox());
      isLoaded = true;
    }
  }

  virtual ~GeometryDomain() {}

  GeometryDomain(const GeometryDomain &other) : AbstractDomain(other) {
    mesh = other.mesh;
    lights = other.lights;
    boundingBox = other.boundingBox;
    filename = other.filename;
    isLoaded = true;
  }

  virtual bool hasGeometry() { return isLoaded; }

  virtual int size() { return 0; }

  virtual int sizeInBytes() { return 0; }

  virtual std::vector<gvt::render::data::scene::Light *> &getLights() { return lights; }

  virtual void setLights(std::vector<gvt::render::data::scene::Light *> &l) { lights = l; }

  virtual gvt::render::data::primitives::Mesh *getMesh() { return mesh; }

  virtual bool load();
  virtual void free();

  friend std::ostream &operator<<(std::ostream &, GeometryDomain const &);

  virtual void operator()() {}

protected:
  gvt::render::data::primitives::Mesh *mesh;
  std::vector<gvt::render::data::scene::Light *> lights;
  std::string filename;
};
}
}
}
}

#endif // GVT_RENDER_DATA_DOMAIN_GEOMETRY_DOMAIN_H
