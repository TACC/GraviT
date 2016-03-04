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

#include <gvt/render/data/domain/AbstractDomain.h>

using namespace gvt::render::data::domain;

AbstractDomain::AbstractDomain(glm::mat4 m) : m(m), domainID(-1), isLoaded(false) {
  // minv = m.inverse();
  // normi = m.upper33().inverse().transpose();
}

AbstractDomain::AbstractDomain(const AbstractDomain &other) {
  // m = other.m;
  // minv = other.minv;
  // normi = other.normi;
}

AbstractDomain::~AbstractDomain() {}

bool AbstractDomain::intersect(gvt::render::actor::Ray &r, gvt::render::actor::isecDomList &inter) {
  float t;
  if (getWorldBoundingBox().intersectDistance(r, t) && t > gvt::render::actor::Ray::RAY_EPSILON) {
    inter.push_back(gvt::render::actor::isecDom(domainID, t));
    return true;
  }
  return false;
}

// TODO : This code assumes non-overlapping domains
void AbstractDomain::marchIn(gvt::render::actor::Ray &ray) {
  gvt::render::data::primitives::Box3D wBox = getWorldBoundingBox();
  float t = FLT_MAX;
  ray.setDirection(-ray.direction);
  while (wBox.inBox(ray.origin)) {
    if (wBox.intersectDistance(ray, t)) ray.origin += ray.direction * t;
    ray.origin += ray.direction * gvt::render::actor::Ray::RAY_EPSILON;
  }
  ray.setDirection(-ray.direction);
}
// TODO : This code assumes non-overlapping domains
void AbstractDomain::marchOut(gvt::render::actor::Ray &ray) {
  gvt::render::data::primitives::Box3D wBox = getWorldBoundingBox();
  float t = FLT_MAX;

  if (wBox.intersectDistance(ray, t)) ray.origin += ray.direction * t;
  while (wBox.intersectDistance(ray, t)) {
    ray.origin += ray.direction * t;
    ray.origin += ray.direction * gvt::render::actor::Ray::RAY_EPSILON;
  }
  ray.origin += ray.direction * gvt::render::actor::Ray::RAY_EPSILON;
}

void AbstractDomain::trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays) {
  GVT_ASSERT(false, "Trace function for this domain was not implemented");
}

bool AbstractDomain::load() {
  GVT_ASSERT(false, "Calling domain load generic function\n");
  return false;
}

void AbstractDomain::free() {
  GVT_WARNING(false, "Calling domain free generic function\n");
  return;
}

gvt::render::actor::Ray AbstractDomain::toLocal(gvt::render::actor::Ray &r) {
  GVT_ASSERT((&r), "NULL POINTER");
  gvt::render::actor::Ray ray(r);
  ray.origin = glm::vec3(minv * glm::vec4(ray.origin, 1.f));
  ray.direction = glm::vec3(minv * glm::vec4(ray.direction, 0.f));
  return ray;
}

gvt::render::actor::Ray AbstractDomain::toWorld(gvt::render::actor::Ray &r) {
  GVT_ASSERT((&r), "NULL POINTER");
  gvt::render::actor::Ray ray(r);
  ray.origin = glm::vec3(m * glm::vec4(ray.origin, 1.f));
  ray.direction = glm::vec3(m * glm::vec4(ray.direction, 0.f));
  return ray;
}

glm::vec3 AbstractDomain::toLocal(const glm::vec3 &r) {
  glm::vec3 ret = glm::vec3(minv * glm::vec4(r, 1.f));
  return ret;
}

glm::vec3 AbstractDomain::toWorld(const glm::vec3 &r) {
  glm::vec3 ret = glm::vec3(m * glm::vec4(r, 1.f));
  return ret;
}

glm::vec3 AbstractDomain::localToWorldNormal(const glm::vec3 &v) {
  glm::vec3 ret = normi * glm::vec3(v);
  return ret;
}

void AbstractDomain::translate(glm::vec3 t) {

   m = glm::translate(m, t);

   minv = glm::inverse(m);

   normi = glm::transpose(glm::inverse(glm::mat3(m)));

}

void AbstractDomain::rotate(glm::vec3 t) {

	m = glm::rotate(m, t[0], glm::vec3(1, 0, 0))
			* glm::rotate(m, t[1], glm::vec3(0, 1, 0))
			* glm::rotate(m, t[2], glm::vec3(0, 0, 1));

	minv = glm::inverse(m);

	normi = glm::transpose(glm::inverse(glm::mat3(m)));

	// m = m * glm::mat4::createRotation(t[0], 1.0, 0.0, 0.0) *
	//     glm::mat4::createRotation(t[1], 0.0, 1.0, 0.0) *
	//     glm::mat4::createRotation(t[2], 0.0, 0.0, 1.0);
	// minv = m.inverse();
	// normi = m.upper33().inverse().transpose();
}

void AbstractDomain::scale(glm::vec3 t) {

	  m = glm::scale(m, t);
	    minv = glm::inverse(m);
	    normi = glm::transpose(glm::inverse(glm::mat3(m)));

}

gvt::render::data::primitives::Box3D AbstractDomain::getWorldBoundingBox() { return getBounds(1); }

void AbstractDomain::setBoundingBox(gvt::render::data::primitives::Box3D bb) { boundingBox = bb; }

gvt::render::data::primitives::Box3D AbstractDomain::getBounds(int type = 0) const {
  if (type == 0) {
    return boundingBox;
  } else {

    gvt::render::data::primitives::Box3D bb;
    bb.bounds[0] = glm::vec3(m * glm::vec4(boundingBox.bounds[0], 1.f));
    bb.bounds[1] = glm::vec3(m * glm::vec4(boundingBox.bounds[1], 1.f));
    return bb;
  }
}

bool AbstractDomain::domainIsLoaded() { return isLoaded; }

int AbstractDomain::getDomainID() { return domainID; }

void AbstractDomain::setDomainID(int id) { domainID = id; }

glm::vec3 AbstractDomain::worldCentroid() const {
  gvt::render::data::primitives::Box3D bbox = getBounds(1);
  return (0.5f * bbox.bounds[0]) + (0.5f * bbox.bounds[1]);
}
