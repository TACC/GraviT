/*
 * File:   gvt_optix.h
 * Author: jbarbosa
 *
 * Created on April 22, 2014, 12:47 PM
 */

#ifndef GVT_OPTIX_H
#define GVT_OPTIX_H

#include <GVT/Math/GVTMath.h>
#include <GVT/Data/transform.h>
#include <GVT/Data/primitives.h>
#include <GVT/Data/primitives/gvt_ray.h>
#include <vector>

namespace GVT {

namespace Data {


struct OptixRayFormat {
  float origin_x;
  float origin_y;
  float origin_z;
  float t_min;
  float direction_x;
  float direction_y;
  float direction_z;
  float t_max;
};

struct OptixHitFormat {
  float t;
  int triangle_id;
  float u;
  float v;
};




}

}




#if 0
namespace GVT {

namespace Data {

typedef GVT::Data::Vector4f OptixVector;
typedef GVT::Data::ray OptixRay;
typedef GVT::Data::Mesh OptixMesh;
typedef std::vector<OptixRay> OptixRayQueue;

template <>
struct transform_impl<GVT::Math::Vector4f, OptixVector> {
  static inline OptixVector transform(const GVT::Math::Vector4f& r) {
    return OptixVector(r[0], r[1], r[2], r[3]);
  }
};

template <>
struct transform_impl<OptixVector, GVT::Math::Vector4f> {
  static inline GVT::Math::Vector4f transform(const OptixVector& r) {
    return GVT::Math::Vector4f(r[0], r[1], r[2], r[3]);
  }
};

template <>
struct transform_impl<GVT::Data::ray, OptixRay> {
  static inline OptixRay transform(const GVT::Data::ray& r) {
    OptixRay ray;
    const OptixVector orig =
        GVT::Data::transform<GVT::Math::Point4f, Manta::Vector>(r.origin);
    const OptixVector dir =
        GVT::Data::transform<GVT::Math::Vector4f, Manta::Vector>(r.direction);
    return OptixRay(orig, dir);
  }
};

template <>
struct transform_impl<OptixRay, GVT::Data::ray> {
  static inline GVT::Data::ray transform(const OptixRay& r) {
    return GVT::Data::ray(
        GVT::Data::transform<OptixVector, GVT::Math::Point4f>(r.origin()),
        GVT::Data::transform<OptixVector, GVT::Math::Vector4f>(r.direction()));
  }
};

}  // namespace Data

}  // namespace GVT
#endif
#endif /* GVT_OPTIX_H */

