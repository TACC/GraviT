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
// Domain.h
//

#ifndef GVT_RENDER_DATA_DOMAIN_ABSTRACT_DOMAIN_H
#define GVT_RENDER_DATA_DOMAIN_ABSTRACT_DOMAIN_H

#include <gvt/render/data/Primitives.h>
#include <gvt/core/Math.h>
#include <vector>

#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/mutex.hpp>

namespace gvt {
namespace render {
namespace data {
namespace domain {
/// base class for data domains
/** base class for GraviT data domains. A domain is the atomic data representation within
GraviT. A domain is passed to a rendering engine in its entirety, where it might be
subdivided for the engine's use. Domains are not subdivided within GraviT.
*/
class AbstractDomain {
protected:
  AbstractDomain(gvt::core::math::AffineTransformMatrix<float> m = gvt::core::math::AffineTransformMatrix<float>(true));
  AbstractDomain(const AbstractDomain &other);
  virtual ~AbstractDomain();

public:
  virtual bool intersect(gvt::render::actor::Ray &r, gvt::render::actor::isecDomList &inter);

  virtual void marchIn(gvt::render::actor::Ray &r);
  virtual void marchOut(gvt::render::actor::Ray &r);
  virtual void trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays);

  virtual bool load();
  virtual void free();
  virtual int size() = 0;
  virtual int sizeInBytes() = 0;

  virtual gvt::render::actor::Ray toLocal(gvt::render::actor::Ray &r);

  virtual gvt::render::actor::Ray toWorld(gvt::render::actor::Ray &r);

  virtual gvt::core::math::Vector4f toLocal(const gvt::core::math::Vector4f &r);

  virtual gvt::core::math::Vector4f toWorld(const gvt::core::math::Vector4f &r);

  virtual gvt::core::math::Vector4f localToWorldNormal(const gvt::core::math::Vector4f &v);
  virtual gvt::render::data::primitives::Box3D getWorldBoundingBox();

  virtual void setBoundingBox(gvt::render::data::primitives::Box3D bb);

  virtual gvt::render::data::primitives::Box3D getBounds(int type) const;

  virtual bool domainIsLoaded();

  virtual int getDomainID();

  virtual void setDomainID(int id);

  virtual void translate(gvt::core::math::Vector4f t);
  virtual void rotate(gvt::core::math::Vector4f t);
  virtual void scale(gvt::core::math::Vector4f t);

  virtual gvt::core::math::Point4f worldCentroid() const;

  // Public variables
  gvt::core::math::AffineTransformMatrix<float> m;
  gvt::core::math::AffineTransformMatrix<float> minv;
  gvt::core::math::Matrix3f normi;
  gvt::render::data::primitives::Box3D boundingBox;

  boost::mutex _inqueue;
  boost::mutex _outqueue;

  int domainID;

  bool isLoaded;
};
}
}
}
}
#endif // GVT_RENDER_DATA_DOMAIN_H
