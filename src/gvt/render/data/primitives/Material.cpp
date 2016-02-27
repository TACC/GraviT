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
 * File:   Material.cpp
 * Author: jbarbosa
 *
 * Created on April 18, 2014, 3:07 PM
 */

#include <cmath>
#include <gvt/render/data/DerivedTypes.h>
#include <gvt/render/data/primitives/Material.h>

using namespace gvt::core::math;
using namespace gvt::render::actor;
using namespace gvt::render::data::primitives;
using namespace gvt::render::data::scene;

Material::Material() {}

Material::Material(const Material &orig) {}

Material::~Material() {}

Vector4f Material::shade(const Ray &ray, const Vector4f &sufaceNormal, const Light *lightSource) { return Vector4f(); }

RayVector Material::ao(const Ray &ray, const Vector4f &sufaceNormal, float samples) { return RayVector(); }

RayVector Material::secondary(const Ray &ray, const Vector4f &sufaceNormal, float samples) { return RayVector(); }

Lambert::Lambert(const Vector4f &kd) : Material(), kd(kd) {}

Lambert::Lambert(const Lambert &orig) : Material(orig), kd(orig.kd) {}

Lambert::~Lambert() {}

Vector4f Lambert::shade(const Ray &ray, const Vector4f &N, const Light *lightSource) {

  Point4f V = ray.direction;
  V = V.normalize();
  float NdotL = std::max(0.f, std::abs(N * V));
  Color lightSourceContrib = lightSource->contribution(ray);
  Color diffuse = prod(lightSourceContrib, kd * NdotL) * ray.w;
  return diffuse;
}

RayVector Lambert::ao(const Ray &ray, const Vector4f &sufaceNormal, float samples) { return RayVector(); }

RayVector Lambert::secundary(const Ray &ray, const Vector4f &sufaceNormal, float samples) { return RayVector(); }

Phong::Phong(const Vector4f &kd, const Vector4f &ks, const float &alpha) : Material(), kd(kd), ks(ks), alpha(alpha) {}

Phong::Phong(const Phong &orig) : Material(orig), kd(orig.kd), ks(orig.ks), alpha(orig.alpha) {}

Phong::~Phong() {}

Vector4f Phong::shade(const Ray &ray, const Vector4f &N, const Light *lightSource) {
  Vector4f hitPoint = (Vector4f)ray.origin + (ray.direction * ray.t);
  Vector4f L = (Vector4f)lightSource->position - hitPoint;

  L = L.normalize();
  float NdotL = std::max(0.f, (N * L));
  Vector4f R = ((N * 2.f) * NdotL) - L;
  float VdotR = std::max(0.f, (R * (-ray.direction)));
  float power = VdotR * std::pow(VdotR, alpha);

  Vector4f lightSourceContrib = lightSource->contribution(ray); //  distance;

  Color diffuse = prod((lightSourceContrib * NdotL), kd) * ray.w;
  Color specular = prod((lightSourceContrib * power), ks) * ray.w;

  Color finalColor = (diffuse + specular);
  return finalColor;
}

RayVector Phong::ao(const Ray &ray, const Vector4f &sufaceNormal, float samples) { return RayVector(); }

RayVector Phong::secundary(const Ray &ray, const Vector4f &sufaceNormal, float samples) { return RayVector(); }

BlinnPhong::BlinnPhong(const Vector4f &kd, const Vector4f &ks, const float &alpha)
    : Material(), kd(kd), ks(ks), alpha(alpha) {}

BlinnPhong::BlinnPhong(const BlinnPhong &orig) : Material(orig), kd(orig.kd), ks(orig.ks), alpha(orig.alpha) {}

BlinnPhong::~BlinnPhong() {}

Vector4f BlinnPhong::shade(const Ray &ray, const Vector4f &N, const Light *lightSource) {
  Vector4f hitPoint = (Vector4f)ray.origin + (ray.direction * ray.t);
  Vector4f L = (Vector4f)lightSource->position - hitPoint;
  L = L.normalize();
  float NdotL = std::max(0.f, (N * L));

  Vector4f H = (L - ray.direction).normalize();

  float NdotH = (H * N);
  float power = NdotH * std::pow(NdotH, alpha);

  Vector4f lightSourceContrib = lightSource->contribution(ray);

  Color diffuse = prod((lightSourceContrib * NdotL), kd) * ray.w;
  Color specular = prod((lightSourceContrib * power), ks) * ray.w;

  Color finalColor = (diffuse + specular);
  return finalColor;
}

RayVector BlinnPhong::ao(const Ray &ray, const Vector4f &sufaceNormal, float samples) { return RayVector(); }

RayVector BlinnPhong::secundary(const Ray &ray, const Vector4f &sufaceNormal, float samples) { return RayVector(); }