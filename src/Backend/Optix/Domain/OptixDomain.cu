#include <string>
#include <boost/timer/timer.hpp>


#include <Backend/Optix/Domain/OptixDomain.h>
#include <GVT/Data/primitives.h>
#include <algorithm>
#include <GVT/common/utils.h>
#include <optix_prime/optix_primepp.h>
#include <Backend/Optix/Data/optix_dataformat.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>


using GVT::Data::Color;
using GVT::Data::LightSource;
using GVT::Data::Mesh;
using GVT::Data::RayVector;
using GVT::Data::ray;
using GVT::Domain::GeometryDomain;
using GVT::Math::Vector3f;
using GVT::Math::Vector4f;
using optix::prime::BufferDesc;
using optix::prime::Context;
using optix::prime::Model;
using optix::prime::Query;

namespace GVT {

namespace Domain {

//struct OptixRayFormat {
//  float origin_x;
//  float origin_y;
//  float origin_z;
//  float t_min;
//  float direction_x;
//  float direction_y;
//  float direction_z;
//  float t_max;
//};
//
//struct OptixHitFormat {
//  float t;
//  int triangle_id;
//  float u;
//  float v;
//};

static void gravityRayToOptixRay(const ray& gvt_ray, OptixRayFormat& optix_ray) {
#if 0
  optix_ray.origin[0] = gvt_ray.origin[0];
  optix_ray.origin[1] = gvt_ray.origin[1];
  optix_ray.origin[2] = gvt_ray.origin[2];
  optix_ray.direction[0] = gvt_ray.direction[0];
  optix_ray.direction[1] = gvt_ray.direction[1];
  optix_ray.direction[2] = gvt_ray.direction[2];
  optix_ray.t_min = gvt_ray.t_min;
  optix_ray.t_max = gvt_ray.t_max;
#endif
}

OptixDomain::OptixDomain() : GeometryDomain("") {}

OptixDomain::OptixDomain(const std::string& filename)
    : GeometryDomain(filename), loaded_(false) {}

OptixDomain::OptixDomain(const OptixDomain& domain)
    : GeometryDomain(domain),
      optix_context_(domain.optix_context_),
      optix_model_(domain.optix_model_),
      loaded_(false) {}

OptixDomain::~OptixDomain() {}

OptixDomain::OptixDomain(const std::string& filename,
                         GVT::Math::AffineTransformMatrix<float> m)
    : GVT::Domain::GeometryDomain(filename, m), loaded_(false) {
  this->load();
}

bool OptixDomain::load() {
#if 0
  if (loaded_) return true;

  // Make sure we load the GVT mesh.
  GVT_ASSERT(GeometryDomain::load(), "Geometry not loaded");
  if (!GeometryDomain::load()) return false;

  // Make sure normals exist.
  if (this->mesh->normals.size() < this->mesh->vertices.size() ||
      this->mesh->face_normals.size() < this->mesh->faces.size())
    this->mesh->generateNormals();

  // Create an Optix context to use.
  optix_context_ = Context::create(RTP_CONTEXT_TYPE_CUDA);

  GVT_ASSERT(optix_context_.isValid(), "Optix Context is not valid");
  if (!optix_context_.isValid()) return false;

  // Setup the buffer to hold our vertices.
  BufferDesc vertices_desc;
  vertices_desc = optix_context_->createBufferDesc(
      RTP_BUFFER_FORMAT_VERTEX_FLOAT3, RTP_BUFFER_TYPE_HOST,
      &this->mesh->vertices[0]);

  GVT_ASSERT(vertices_desc.isValid(), "Vertices are not valid");
  if (!vertices_desc.isValid()) return false;

  vertices_desc->setRange(0, this->mesh->vertices.size());
  vertices_desc->setStride(sizeof(Vector3f));

  // Setup the triangle indices buffer.
  BufferDesc indices_desc;
  indices_desc = optix_context_->createBufferDesc(
      RTP_BUFFER_FORMAT_INDICES_INT3, RTP_BUFFER_TYPE_HOST,
      &this->mesh->faces[0]);

  GVT_ASSERT(indices_desc.isValid(), "Indices are not valid");
  if (!indices_desc.isValid()) return false;

  indices_desc->setRange(0, this->mesh->faces.size());
  indices_desc->setStride(sizeof(Mesh::face));

  // Create an Optix model.
  optix_model_ = optix_context_->createModel();

  GVT_ASSERT(optix_model_.isValid(), "Model is not valid");
  if (!optix_model_.isValid()) return false;

  optix_model_->setTriangles(indices_desc, vertices_desc);
  optix_model_->update(RTP_MODEL_HINT_NONE);
  optix_model_->finish();

  if (!optix_model_.isValid()) return false;

  loaded_ = true;
#endif
  return true;
}

void OptixDomain::trace(RayVector& ray_list, RayVector& moved_rays) {
  // Create our query.
  boost::timer::auto_cpu_timer optix_time("Optix domain trace: %w\n");
#if 0    
  try {
    this->load();
    
    
    if (!this->mesh->haveNormals || this->mesh->normals.size() == 0)
      this->mesh->generateNormals();

    GVT_ASSERT(optix_model_.isValid(), "trace:Model is not valid");
    if (!optix_model_.isValid()) return;

    
    // Converting the list why???
    RayVector next_list;
    while (!ray_list.empty()) {
      next_list.push_back(ray_list.back());
      ray_list.pop_back();
    }
    
    
    
    RayVector chunk;  
    const uint32_t kMaxChunkSize = 65536;
    
    chunk.reserve(kMaxChunkSize);
    
    while (!next_list.empty()) {
                
      ray_list.swap(next_list);
      
      while (!ray_list.empty()) {
        chunk.push_back(ray_list.back());
        ray_list.pop_back();
      
        if (chunk.size() == kMaxChunkSize || ray_list.empty()) {
          traceChunk(chunk, next_list, moved_rays);
          chunk.clear();
        }
    
      }
    
      //std::cout << "Next rays : " <<  next_list.size() << std::endl;
    
    }
    
    
  }
  catch (const optix::prime::Exception& e) {
    GVT_ASSERT(false, e.getErrorString());
  }
#endif
}

void OptixDomain::traceChunk(RayVector& chunk, RayVector& next_list, RayVector& moved_rays) {
#if 0
    // Create our query.
  Query query = optix_model_->createQuery(RTP_QUERY_TYPE_CLOSEST);
  if (!query.isValid()) return;

  // Format GVT rays for Optix and give Optix an array of rays.
  std::vector<OptixRayFormat> optix_rays(chunk.size());
  
  
  
  for (int i = 0; i < chunk.size(); ++i)
    gravityRayToOptixRay(this->toLocal(chunk[i]), optix_rays[i]);

  // Hand the rays to Optix.
  query->setRays(optix_rays.size(),
                 RTP_BUFFER_FORMAT_RAY_ORIGIN_TMIN_DIRECTION_TMAX,
                 RTP_BUFFER_TYPE_HOST, &optix_rays[0]);

  // Create and pass hit results in an Optix friendly format.
  std::vector<OptixHitFormat> hits(chunk.size());
  query->setHits(hits.size(), RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V,
                 RTP_BUFFER_TYPE_HOST, &hits[0]);

  // Execute our query and wait for it to finish.
  query->execute(RTP_QUERY_HINT_NONE);
  query->finish();

  // Move missed rays.
  for (int i = hits.size() - 1; i >= 0; --i) {
    // std::cout << "triangle_id = " << hits[i].triangle_id << "\n";
    if (hits[i].triangle_id < 0) {
      moved_rays.push_back(chunk[i]);
      std::swap(hits[i], hits.back());
      std::swap(chunk[i], chunk.back());
      chunk.pop_back();
      hits.pop_back();
    }
  }

  //std::cout << "num hits = " << hits.size() << "\n";
  // Trace each ray: shade, fire shadow rays, fire secondary rays.
  for (int i = 0; i < chunk.size(); ++i)
    this->traceRay(hits[i].triangle_id, hits[i].t, hits[i].u, hits[i].v, chunk[i], next_list);
#endif
}

void OptixDomain::traceRay(uint32_t triangle_id, float t, float u, float v, ray& ray, RayVector& rays) {
#if 0
    if (ray.type == ray::SHADOW) return;
  if (ray.type == ray::SECONDARY) {
    float s = ((t > 1.0f) ? 1.0f / t : t);
    ray.w = ray.w * s;
    //std::cout << "w = " << ray.w << "\n";
  }
  ray.t = t;
  Vector4f normal = this->localToWorldNormal(computeNormal(triangle_id, u, v));
  normal.normalize();
  generateShadowRays(triangle_id, ray, normal, rays);
  generateSecondaryRays(ray, normal, rays);
#endif 
}

Vector4f OptixDomain::computeNormal(uint32_t triangle_id, float u, float v) const {
#if 0
  const Mesh::face_to_normals& normals =
      this->mesh->faces_to_normals[triangle_id];
  const Vector4f& a = this->mesh->normals[normals.get<0>()];
  const Vector4f& b = this->mesh->normals[normals.get<1>()];
  const Vector4f& c = this->mesh->normals[normals.get<2>()];
  Vector4f normal = a * u + b * v + c * (1.0f - u - v);
  normal.normalize();
  return normal;
#endif
}

void OptixDomain::generateSecondaryRays(const ray& ray_in,
                                        const Vector4f& normal,
                                        RayVector& rays) {
#if 0
  int depth = ray_in.depth - 1;
  float p = 1.0f - (float(rand()) / RAND_MAX);
  if (depth > 0 && ray_in.w > p) {
    ray secondary_ray(ray_in);
    secondary_ray.domains.clear();
    secondary_ray.type = ray::SECONDARY;
    // Try to ensure that the shadow ray is on the correct side of the triangle.
    // Technique adapted from "Robust BVH Ray Traversal" by Thiago Ize.
    // Using about 8 * ULP(t).
    float multiplier = 1.0f - 16.0f * std::numeric_limits<float>::epsilon();
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
#endif
}

void OptixDomain::generateShadowRays(int triangle_id, const ray& ray_in,
                                     const Vector4f& normal, RayVector& rays) {
#if 0
  for (int lindex = 0; lindex < this->lights.size(); ++lindex) {
    ray shadow_ray(ray_in);
    shadow_ray.domains.clear();
    shadow_ray.type = ray::SHADOW;
    // Try to ensure that the shadow ray is on the correct side of the triangle.
    // Technique adapted from "Robust BVH Ray Traversal" by Thiago Ize.
    // Using about 8 * ULP(t).
    float multiplier = 1.0f - 16.0f * std::numeric_limits<float>::epsilon();
    float t_shadow = multiplier * shadow_ray.t;
    shadow_ray.origin = shadow_ray.origin + shadow_ray.direction * t_shadow;
    Vector4f light_position(this->lights[lindex]->position);
    Vector4f dir = light_position - shadow_ray.origin;
    shadow_ray.t_max = dir.length();
    dir.normalize();
    shadow_ray.setDirection(dir);
    Color c = this->mesh->shadeFace(triangle_id, shadow_ray, normal,
                                this->lights[lindex]);
    //Color c = this->mesh->mat->shade(shadow_ray, normal, this->lights[lindex]);
    shadow_ray.color = COLOR_ACCUM(1.0f, c[0], c[1], c[2], 1.0f);
    rays.push_back(shadow_ray);
  }
#endif
}

}  // namespace Domain

}  // namespace GVT
