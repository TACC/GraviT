

//#include "cutil_math.h"
//#include "Material.cuh"

#include <gvt/render/adapter/optix/data/Material.cuh>

#include "cutil_math.h"

using namespace gvt::render::data::cuda_primitives;
using namespace gvt::render::data::primitives;

__device__ inline float4 toFloat4(glm::vec3 v){
	return make_float4(v.x, v.y,v.z, 0.f);
}

   __device__ float4 lambertShade(const gvt::render::data::primitives::Material* material,
       const Ray &ray, const float4 &N,const float4& wi) {


    float NdotL = fmaxf(0.f, (N * wi));
    float4 diffuse = toFloat4(material->kd) *  (NdotL * ray.w);
    return diffuse;
  }


   __device__ float4 phongShade(const gvt::render::data::primitives::Material* material,
                   const Ray &ray, const float4 &N,const float4& wi) {


    float NdotL = fmaxf(0.f, (N * wi));
    float4 R = ((N * 2.f) * NdotL) - wi;
    float VdotR = max(0.f, (R * (-1*ray.direction)));
    float power = VdotR * pow(VdotR, material->alpha);


    float4 diffuse = NdotL * toFloat4(material->kd) * ray.w;
    float4 specular = power * toFloat4(material->ks) * ray.w;

    float4 finalColor = (diffuse + specular);
    return finalColor;
  }

   __device__ float4 blinnShade(const gvt::render::data::primitives::Material* material,
                   const Ray &ray, const float4 &N, const float4& wi) {

     float NdotL = fmaxf(0.f, (N * wi));

    float4 H = normalize((wi - ray.direction));

    float NdotH = (H * N);
    float power = NdotH * pow(NdotH, material->alpha);

    float4 diffuse = NdotL * toFloat4(material->kd) * ray.w;
    float4 specular = power * toFloat4(material->ks) * ray.w;

    float4 finalColor = (diffuse + specular);
    return finalColor;
  }


  __device__ bool gvt::render::data::cuda_primitives::Shade(gvt::render::data::primitives::Material *material,
                                                          const Ray &ray, const float4 &sufaceNormal,
                                                          const Light *lightSource, const float4 lightPosSample,
                                                          float4 &r) {

  float4 hitPoint = ray.origin + ray.direction * ray.t;
  float4 wi = normalize(lightPosSample - hitPoint);
  float NdotL = fmaxf(0.f, (sufaceNormal * wi));
  float4 Li = lightSource->contribution(hitPoint, lightPosSample);

  if (NdotL == 0.f || (Li.x == 0.f && Li.y == 0.f && Li.z == 0.f)) return false;

  switch (material->type) {
  case LAMBERT:
    r = lambertShade(material, ray, sufaceNormal, wi);
    break;
  case PHONG:
    r = phongShade(material, ray, sufaceNormal, wi);
    break;
  case BLINN:
    r = blinnShade(material, ray, sufaceNormal, wi);
    break;
  default:
    printf("Material implementation missing for cuda-optix adpater\n");
    break;
  }

  r*=Li;

  return true;
};
