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
#include <gvt/render/data/Dataset.h>
#include <gvt/render/Attributes.h>
#include <gvt/render/data/accel/BVH.h>

#include <boost/range/algorithm.hpp>
#include <boost/foreach.hpp>

using namespace gvt::render::data;
using namespace gvt::render::data::domain;
using namespace gvt::render::data::primitives;
using namespace gvt::render::data::scene;
using namespace gvt::render::data::accel;
using namespace gvt::render::actor;

Dataset::Dataset() : acceleration(NULL) {}

Dataset::~Dataset() {}

bool Dataset::init() {
  GVT_DEBUG(DBG_ALWAYS, "Dataset::init() abstract load");
  return false;
}

bool Dataset::intersect(Ray &r, isecDomList &inter) {
  if (dataSetBB.intersect(r) || dataSetBB.inBox(r)) {
    r.t = FLT_MAX;
    if (!acceleration) {
      // clang-format off
      BOOST_FOREACH(AbstractDomain * d, domainSet)
      d->intersect(r, inter);
      // clang-format on
    } else {
      acceleration->intersect(r, inter);
    }
    boost::sort(inter);
    return (!inter.empty());
  }
  return false;
}

AbstractDomain *Dataset::getDomain(size_t id) {
  GVT_ASSERT_BACKTRACE(id < domainSet.size(), "Getting domain outside bound");
  return domainSet[id];
}

Light *Dataset::getLight(size_t id) {
  GVT_ASSERT_BACKTRACE(id < lightSet.size(), "Getting light source outside bound");
  return lightSet[id];
}

int Dataset::addDomain(AbstractDomain *dom) {
  dataSetBB.merge(dom->getWorldBoundingBox());
  domainSet.push_back(dom);
  dom->setDomainID(domainSet.size() - 1);

  GVT_DEBUG(DBG_ALWAYS, "Add domain. World BB : " << dataSetBB);

  return domainSet.size() - 1;
}

int Dataset::addLight(Light *ls) {
  dataSetBB.merge(ls->getWorldBoundingBox());
  lightSet.push_back(ls);
  return domainSet.size() - 1;
}

int Dataset::size() { return domainSet.size(); }

void Dataset::makeAccel() {}
void Dataset::makeAccel(gvt::render::Attributes &rta) {}
