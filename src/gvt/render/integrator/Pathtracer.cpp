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
/*
 * PathtracerShader.cpp
 *
 *  Created on: Mar 6, 2016
 *      Author: Roberto Ribeiro
 */

#include <gvt/render/integrator/Pathtracer.h>
#include <gvt/render/actor/Ray.h>
#include <gvt/render/data/DerivedTypes.h>
#include <gvt/render/data/primitives/Material.h>


using namespace gvt::render;

Pathtracer::Pathtracer(
		const std::vector<gvt::render::data::scene::Light *>& lights) : Integrator(), lights(lights){

}

Pathtracer::~Pathtracer(){

}

glm::vec3 Pathtracer::CosWeightedRandomHemisphereDirection2(glm::vec3 n) {
  float Xi1 = (float)rand() / (float)RAND_MAX;
  float Xi2 = (float)rand() / (float)RAND_MAX;

  float theta = std::acos(std::sqrt(1.0 - Xi1));
  float phi = 2.0 * 3.1415926535897932384626433832795 * Xi2;

  float xs = sinf(theta) * cosf(phi);
  float ys = cosf(theta);
  float zs = sinf(theta) * sinf(phi);

  glm::vec3 y(n);
  glm::vec3 h = y;
  if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z))
    h[0] = 1.0;
  else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z))
    h[1] = 1.0;
  else
    h[2] = 1.0;

  glm::vec3 x = glm::cross(h, y);
  glm::vec3 z = glm::cross(x, y);

  glm::vec3 direction = x * xs + y * ys + z * zs;
  return glm::normalize(direction);
}

void Pathtracer::generateShadowRays(const gvt::render::actor::Ray &r, const glm::vec3 &normal,
                 gvt::render::data::primitives::Material * material, unsigned int *randSeed,
                        gvt::render::actor::RayVector& shadowRays) {

  for (gvt::render::data::scene::Light *light : lights) {
    GVT_ASSERT(light, "generateShadowRays: light is null for some reason");
    // Try to ensure that the shadow ray is on the correct side of the
    // triangle.
    // Technique adapted from "Robust BVH Ray Traversal" by Thiago Ize.
    // Using about 8 * ULP(t).

    gvt::render::data::Color c;
    glm::vec3 lightPos;
    if (light->LightT == gvt::render::data::scene::Light::Area) {
      lightPos = ((gvt::render::data::scene::AreaLight *)light)->GetPosition(randSeed);
    } else {
      lightPos = light->position;
    }

    //c = material->shade(r, normal, light, lightPos);
    //AMaterial m;

    c = gvt::render::data::primitives::Shade(
          material, r, normal, light, lightPos);

    const float multiplier = 1.0f - 16.0f * std::numeric_limits<float>::epsilon();
    const float t_shadow = multiplier * r.t;

    const glm::vec3 origin = r.origin + r.direction * t_shadow;
    const glm::vec3 dir = light->position - origin;
    const float t_max = dir.length();

    // note: ray copy constructor is too heavy, so going to build it manually
    shadowRays.push_back(Ray(r.origin + r.direction * t_shadow, dir, r.w, Ray::SHADOW, r.depth));

    Ray &shadow_ray = shadowRays.back();
    shadow_ray.t = r.t;
    shadow_ray.id = r.id;
    shadow_ray.t_max = t_max;

    // gvt::render::data::Color c = adapter->getMesh()->mat->shade(shadow_ray,
    // normal, lights[lindex]);
    shadow_ray.color = GVT_COLOR_ACCUM(1.0f, c[0], c[1], c[2], 1.0f);
  }
}

bool Pathtracer::L(Ray& r, const glm::vec3 &normal,
		   gvt::render::data::primitives::Material * material,
		TLRand& randEngine, gvt::render::actor::RayVector& shadowRays, int* valid){

	bool validRayLeft = false;
	float t = r.t;
    // reduce contribution of the color that the shadow rays get
    if (r.type == gvt::render::actor::Ray::SECONDARY) {
      t = (t > 1) ? 1.f / t : t;
      r.w = r.w * t;
    }
    //unsigned int *randSeed = randEngine.ReturnSeed();

    generateShadowRays(r, normal,material, randEngine.ReturnSeed(), shadowRays);

    int ndepth = r.depth - 1;

    float p = 1.f - randEngine.fastrand(0, 1); //(float(rand()) / RAND_MAX);
    // replace current ray with generated secondary ray
    if (ndepth > 0 && r.w > p) {
      r.type = gvt::render::actor::Ray::SECONDARY;
      const float multiplier =
          1.0f - 16.0f * std::numeric_limits<float>::epsilon(); // TODO: move out somewhere / make static
      const float t_secondary = multiplier * r.t;
      r.origin = r.origin + r.direction * t_secondary;

      // TODO: remove this dependency on mesh, store material object in the database
      // r.setDirection(adapter->getMesh()->getMaterial()->CosWeightedRandomHemisphereDirection2(normal));
      r.setDirection(CosWeightedRandomHemisphereDirection2(normal));

      r.w = r.w * glm::dot(r.direction, normal);
      r.depth = ndepth;
      validRayLeft = true; // we still have a valid ray in the packet to trace
    } else {
      // secondary ray is terminated, so disable its valid bit
      *valid = 0;
    }

    return validRayLeft;

}
