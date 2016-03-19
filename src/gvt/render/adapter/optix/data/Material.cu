

//#include "cutil_math.h"
//#include "Material.cuh"

#include <gvt/render/data/primitives/Material.h>
#include <gvt/render/adapter/optix/data/CUDAMaterial.h>
#include <gvt/render/adapter/optix/data/Material.cuh>

#include "cutil_math.h"

using namespace gvt::render::data::cuda_primitives;
using namespace gvt::render::data::primitives;

   __device__ float4 lambertShade(const gvt::render::data::primitives::CUDALambert* material,
       const Ray &ray, const float4 &N, const Light *lightSource) {

  float4 V = ray.direction;
    V = normalize(V);
    float NdotL = fmaxf(0.f, std::abs(N * V));
    float4 lightSourceContrib = lightSource->contribution(ray);
    float4 diffuse = prod(lightSourceContrib, material->kd * NdotL) * ray.w;
    return diffuse;
  }


//   __device__ float4 Phong::shade(const Ray &ray, const float4 &N, const Light *lightSource) {
//    float4 hitPoint = (float4)ray.origin + (ray.direction * ray.t);
//    float4 L = (float4)lightSource->light.position - hitPoint;

//    L = normalize(L);
//    float NdotL = fmaxf(0.f, (N * L));
//    float4 R = ((N * 2.f) * NdotL) - L;
//    float VdotR = max(0.f, (R * (-1*ray.direction)));
//    float power = VdotR * std::pow(VdotR, alpha);

//    float4 lightSourceContrib = lightSource->contribution(ray); //  distance;

//    float4 diffuse = prod((lightSourceContrib * NdotL), kd) * ray.w;
//    float4 specular = prod((lightSourceContrib * power), ks) * ray.w;

//    float4 finalColor = (diffuse + specular);
//    return finalColor;
//  }

//   __device__ float4 BlinnPhong::shade(const Ray &ray, const float4 &N, const Light *lightSource) {
//    float4 hitPoint = (float4)ray.origin + (ray.direction * ray.t);
//    float4 L = (float4)lightSource->light.position - hitPoint;
//    L = normalize(L);
//    float NdotL = fmaxf(0.f, (N * L));

//    float4 H = normalize((L - ray.direction));

//    float NdotH = (H * N);
//    float power = NdotH * std::pow(NdotH, alpha);

//    float4 lightSourceContrib = lightSource->contribution(ray);

//    float4 diffuse = prod((lightSourceContrib * NdotL), kd) * ray.w;
//    float4 specular = prod((lightSourceContrib * power), ks) * ray.w;

//    float4 finalColor = (diffuse + specular);
//    return finalColor;
//  }


   __device__  float4 gvt::render::data::cuda_primitives::Shade(
          gvt::render::data::primitives::Material* material,
                             const Ray &ray,
                             const float4 &sufaceNormal,
                             const Light *lightSource,
                             const float4 lightPostion)
     {

                 float4 r;
                   switch (material->type) {
                   case CUDA_LAMBERT:
                           r = lambertShade((CUDALambert*)material,
                                            ray, sufaceNormal, lightSource);
                           break;
//                   case CUDA_PHONG:
//                           r = phong.shade(ray, sufaceNormal, lightSource);
//                           break;
//                   case CUDA_BLINN:
//                           r = blinn.shade(ray, sufaceNormal, lightSource);
//                           break;
                   default:
                           break;
                   }
                 return r;


   };
