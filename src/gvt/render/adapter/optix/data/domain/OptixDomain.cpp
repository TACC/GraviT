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
#include <algorithm>
#include <string>
#include <boost/timer/timer.hpp>

#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>
#include <gvt/core/schedule/TaskScheduling.h>
#include <gvt/render/actor/Ray.h>
#include <gvt/render/adapter/manta/data/Transforms.h>
#include <gvt/render/data/scene/ColorAccumulator.h>
#include <gvt/render/data/scene/Light.h>

#include <gvt/render/adapter/optix/data/domain/OptixDomain.h>
#include <gvt/render/adapter/optix/data/Formats.h>
#include <gvt/render/data/Primitives.h>

//#include <GVT/common/utils.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_prime/optix_primepp.h>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/generate.h>
//#include <thrust/sort.h>
//#include <thrust/copy.h>

using gvt::core::math::Vector3f;
using gvt::core::math::Vector4f;
using gvt::core::math::AffineTransformMatrix;
using gvt::render::actor::Ray;
using gvt::render::actor::RayVector;
using gvt::render::adapter::optix::data::OptixHit;
using gvt::render::adapter::optix::data::OptixRay;
using gvt::render::data::domain::GeometryDomain;
using gvt::render::data::scene::Light;
using gvt::render::data::primitives::Mesh;
using gvt::render::data::Color;

using namespace gvt::render::actor;
using namespace gvt::render::adapter::optix::data;
using namespace gvt::render::adapter::optix::data::domain;
using namespace gvt::render::data::domain;
using namespace gvt::render::data::primitives;

// using optix::prime::BufferDesc;
// using optix::prime::Exception;
// using optix::prime::Context;
// using optix::prime::Model;
// using optix::prime::Query;

static void gvtRayToOptixRay(const Ray &gvt_ray, OptixRay &optix_ray) {
  optix_ray.origin[0] = gvt_ray.origin[0];
  optix_ray.origin[1] = gvt_ray.origin[1];
  optix_ray.origin[2] = gvt_ray.origin[2];
  optix_ray.direction[0] = gvt_ray.direction[0];
  optix_ray.direction[1] = gvt_ray.direction[1];
  optix_ray.direction[2] = gvt_ray.direction[2];
  // optix_ray.t_min = FLT_MAX;  // gvt_ray.t_min;
  // optix_ray.t_max = -FLT_MAX; // gvt_ray.t_max;
}

OptixDomain::OptixDomain() : GeometryDomain("") {}

OptixDomain::OptixDomain(const std::string &filename)
    : GeometryDomain(filename), loaded_(false) {}

OptixDomain::OptixDomain(const OptixDomain &domain)
    : GeometryDomain(domain),
      optix_context_(domain.optix_context_),
      optix_model_(domain.optix_model_),
      loaded_(false) {}

OptixDomain::OptixDomain(GeometryDomain *domain)
    : GeometryDomain(*domain), loaded_(false) {
  this->load();
}

OptixDomain::~OptixDomain() {}

OptixDomain::OptixDomain(const std::string &filename,
                         AffineTransformMatrix<float> m)
    : GeometryDomain(filename, m), loaded_(false) {
  this->load();
}

bool OptixDomain::load() {
  if (loaded_) return true;

  // Make sure we load the GVT mesh.
  GVT_ASSERT(GeometryDomain::load(), "Geometry not loaded");
  if (!GeometryDomain::load()) return false;

  GVT_ASSERT(this->mesh->vertices.size() > 0, "Invalid vertices");
  GVT_ASSERT(this->mesh->faces.size() > 0, "Invalid faces");

  // Make sure normals exist.

  if (this->mesh->normals.size() < this->mesh->vertices.size() ||
      this->mesh->face_normals.size() < this->mesh->faces.size())
    this->mesh->generateNormals();

  // Create an Optix context to use.
  optix_context_ = ::optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);
  GVT_ASSERT(optix_context_.isValid(), "Optix Context is not valid");
  if (!optix_context_.isValid()) return false;

  std::vector<unsigned> activeDevices;
  int devCount = 0;
  cudaDeviceProp prop;
  cudaGetDeviceCount(&devCount);

  GVT_ASSERT(
      devCount,
      "You choose optix render, but no cuda capable devices are present");

  for (int i = 0; i < devCount; i++) {
    cudaGetDeviceProperties(&prop, i);
    if (prop.kernelExecTimeoutEnabled == 0) activeDevices.push_back(i);
  }

  if (!activeDevices.size()) {
    activeDevices.push_back(0);
  }

  optix_context_->setCudaDeviceNumbers(activeDevices);

  // Setup the buffer to hold our vertices.
  //
  std::vector<float> vertices;
  std::vector<int> faces;

  for (auto v : this->mesh->vertices) {
    vertices.push_back(v[0]);
    vertices.push_back(v[1]);
    vertices.push_back(v[2]);
  }
  for (auto f : this->mesh->faces) {
    faces.push_back(f.get<0>());
    faces.push_back(f.get<1>());
    faces.push_back(f.get<2>());
  }
  ::optix::prime::BufferDesc vertices_desc;
  vertices_desc = optix_context_->createBufferDesc(
      RTP_BUFFER_FORMAT_VERTEX_FLOAT3, RTP_BUFFER_TYPE_HOST, &vertices[0]);

  GVT_ASSERT(vertices_desc.isValid(), "Vertices are not valid");
  if (!vertices_desc.isValid()) return false;

  vertices_desc->setRange(0, vertices.size()/3);
  vertices_desc->setStride(sizeof(float) * 3);

  // Setup the triangle indices buffer.
  ::optix::prime::BufferDesc indices_desc;
  indices_desc = optix_context_->createBufferDesc(
      RTP_BUFFER_FORMAT_INDICES_INT3, RTP_BUFFER_TYPE_HOST, &faces[0]);

  GVT_ASSERT(indices_desc.isValid(), "Indices are not valid");
  if (!indices_desc.isValid()) return false;

  indices_desc->setRange(0, faces.size()/3);
  indices_desc->setStride(sizeof(int) * 3);

  // Create an Optix model.
  optix_model_ = optix_context_->createModel();

  GVT_ASSERT(optix_model_.isValid(), "Model is not valid");
  if (!optix_model_.isValid()) return false;

  optix_model_->setTriangles(indices_desc, vertices_desc);
  optix_model_->update(RTP_MODEL_HINT_ASYNC);
  optix_model_->finish();

  if (!optix_model_.isValid()) return false;

  loaded_ = true;
  return true;
}

void OptixDomain::trace(RayVector &ray_list, RayVector &moved_rays) {
  // Create our query.
  boost::timer::auto_cpu_timer optix_time("Optix domain trace: %t\n");
  try {
    this->load();

    if (!this->mesh->haveNormals || this->mesh->normals.size() == 0)
      this->mesh->generateNormals();

    GVT_ASSERT(optix_model_.isValid(), "trace:Model is not valid");
    if (!optix_model_.isValid()) return;
    RayVector chunk;
    while (!ray_list.empty()) {
      chunk.clear();
      chunk.swap(ray_list);
      traceChunk(chunk, ray_list, moved_rays);
    }
  } catch (const std::exception &e) {
    GVT_ASSERT(false, e.what());
  }
}

void OptixDomain::traceChunk(RayVector &chunk, RayVector &next_list,
                             RayVector &moved_rays) {
  // Create our query.
  ::optix::prime::Query query = optix_model_->createQuery(RTP_QUERY_TYPE_CLOSEST);
  if (!query.isValid()) return;

  // Format GVT rays for Optix and give Optix an array of rays.
  std::vector<OptixRay> optix_rays(chunk.size());
  std::vector<OptixHit> hits(chunk.size());
  {
    // boost::timer::auto_cpu_timer optix_time("Convert from GVT to OptiX:
    // %t\n");
    for (int i = 0; i < chunk.size(); ++i) {
      Ray gvt_ray = toLocal(chunk[i]);
      optix_rays[i].origin[0] = gvt_ray.origin[0];
      optix_rays[i].origin[1] = gvt_ray.origin[1];
      optix_rays[i].origin[2] = gvt_ray.origin[2];
      optix_rays[i].direction[0] = gvt_ray.direction[0];
      optix_rays[i].direction[1] = gvt_ray.direction[1];
      optix_rays[i].direction[2] = gvt_ray.direction[2];
      optix_rays[i].t_min = FLT_EPSILON;
      optix_rays[i].t_max = FLT_MAX;
    }
  }

  {
    // boost::timer::auto_cpu_timer optix_time("OptiX intersect call: %t\n");
    // Hand the rays to Optix.
    query->setRays(optix_rays.size(),
                   RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX,
                   RTP_BUFFER_TYPE_HOST, &optix_rays[0]);

    // Create and pass hit results in an Optix friendly format.
    query->setHits(hits.size(), RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V,
                   RTP_BUFFER_TYPE_HOST, &hits[0]);

    // Execute our query and wait for it to finish.
    query->execute(RTP_QUERY_HINT_ASYNC);
    query->finish();
    GVT_ASSERT(query.isValid(), "Something went wrong.");
  }
  // Move missed rays.
  {
    // boost::timer::auto_cpu_timer optix_time("Put rays on outbox queue:
    // %t\n");
    for (int i = hits.size() - 1; i >= 0; --i) {
      if (hits[i].triangle_id < 0) {
        moved_rays.push_back(chunk[i]);
        std::swap(hits[i], hits.back());
        std::swap(chunk[i], chunk.back());
        chunk.pop_back();
        hits.pop_back();
      }
    }
  }

  // std::cout << "num hits = " << hits.size() << "\n";
  // Trace each ray: shade, fire shadow rays, fire secondary rays.
  {
    // boost::timer::auto_cpu_timer optix_time("Try to shade: %t\n");
    for (int i = 0; i < chunk.size(); ++i) {
      if (chunk[i].type == Ray::SHADOW) return;
      if (chunk[i].type == Ray::SECONDARY) {
        float s = ((hits[i].t > 1.0f) ? 1.0f / hits[i].t : hits[i].t);
        chunk[i].w = chunk[i].w * s;
      }
      chunk[i].t = hits[i].t;
      Vector4f normal = this->localToWorldNormal(
          computeNormal(hits[i].triangle_id, hits[i].u, hits[i].v));
      normal.normalize();
      generateShadowRays(hits[i].triangle_id, chunk[i], normal, next_list);
      generateSecondaryRays(chunk[i], normal, next_list);
    }
  }
}

void OptixDomain::traceRay(uint32_t triangle_id, float t, float u, float v,
                           Ray &ray, RayVector &rays) {
  if (ray.type == Ray::SHADOW) return;
  if (ray.type == Ray::SECONDARY) {
    float s = ((t > 1.0f) ? 1.0f / t : t);
    ray.w = ray.w * s;
  }
  ray.t = t;
  Vector4f normal = this->localToWorldNormal(computeNormal(triangle_id, u, v));
  normal.normalize();
  generateShadowRays(triangle_id, ray, normal, rays);
  generateSecondaryRays(ray, normal, rays);
}

Vector4f OptixDomain::computeNormal(const uint32_t &triangle_id, const float &u,
                                    const float &v) const {
  const Mesh::FaceToNormals &normals =
      this->mesh->faces_to_normals[triangle_id];
  const Vector4f &a = this->mesh->normals[normals.get<0>()];
  const Vector4f &b = this->mesh->normals[normals.get<1>()];
  const Vector4f &c = this->mesh->normals[normals.get<2>()];
  Vector4f normal = a * u + b * v + c * (1.0f - u - v);
  normal.normalize();
  return normal;
}

void OptixDomain::generateSecondaryRays(const Ray &ray_in,
                                        const Vector4f &normal,
                                        RayVector &rays) {
  int depth = ray_in.depth - 1;
  float p = 1.0f - (float(rand()) / RAND_MAX);
  if (depth > 0 && ray_in.w > p) {
    Ray secondary_ray(ray_in);
    secondary_ray.domains.clear();
    secondary_ray.type = Ray::SECONDARY;
    // Try to ensure that the shadow ray is on the correct side of the triangle.
    // Technique adapted from "Robust BVH Ray Traversal" by Thiago Ize.
    // Using about 8 * ULP(t).
    float t_secondary = multiplier * secondary_ray.t;
    secondary_ray.origin =
        secondary_ray.origin + secondary_ray.direction * t_secondary;
    secondary_ray.setDirection(
        this->mesh->mat->CosWeightedRandomHemisphereDirection2(normal)
            .normalize());
    secondary_ray.w = secondary_ray.w * (secondary_ray.direction * normal);
    secondary_ray.depth = depth;
    rays.push_back(secondary_ray);
  }
}

void OptixDomain::generateShadowRays(const int &triangle_id, const Ray &ray_in,
                                     const Vector4f &normal, RayVector &rays) {
  for (int lindex = 0; lindex < this->lights.size(); ++lindex) {
    Ray shadow_ray(ray_in);
    shadow_ray.domains.clear();
    shadow_ray.type = Ray::SHADOW;
    // Try to ensure that the shadow ray is on the correct side of the triangle.
    // Technique adapted from "Robust BVH Ray Traversal" by Thiago Ize.
    // Using about 8 * ULP(t).
    float t_shadow = multiplier * shadow_ray.t;
    shadow_ray.origin = shadow_ray.origin + shadow_ray.direction * t_shadow;
    Vector4f light_position(this->lights[lindex]->position);
    Vector4f dir = light_position - shadow_ray.origin;
    shadow_ray.t_max = dir.length();
    dir.normalize();
    shadow_ray.setDirection(dir);
    Color c = this->mesh->shadeFace(triangle_id, shadow_ray, normal,
                                    this->lights[lindex]);
    // Color c = this->mesh->mat->shade(shadow_ray, normal,
    // this->lights[lindex]);
    shadow_ray.color = GVT_COLOR_ACCUM(1.0f, c[0], c[1], c[2], 1.0f);
    rays.push_back(shadow_ray);
    GVT_DEBUG(DBG_ALWAYS, "SHADE_FACE");
  }
}
