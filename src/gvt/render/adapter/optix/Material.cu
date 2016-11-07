

#include <gvt/render/adapter/optix/Material.cuh>

#include "cutil_math.h"

using namespace gvt::render::data::cuda_primitives;
using namespace gvt::render::data::primitives;

__device__ inline cuda_vec tocuda_vec(const glm::vec3& v){
	return make_cuda_vec(v.x, v.y,v.z);
}

   __device__ cuda_vec lambertShade(const gvt::render::data::primitives::Material* material,
       const Ray &ray, const cuda_vec &N,const cuda_vec& wi) {


    float NdotL = fmaxf(0.f, (N * wi));
    cuda_vec diffuse = tocuda_vec(material->kd) *  (NdotL * ray.w);
    return diffuse;
  }


   __device__ cuda_vec phongShade(const gvt::render::data::primitives::Material* material,
                   const Ray &ray, const cuda_vec &N,const cuda_vec& wi) {


    float NdotL = fmaxf(0.f, (N * wi));
    cuda_vec R = ((N * 2.f) * NdotL) - wi;
    float VdotR = max(0.f, (R * (-1*ray.direction)));
    float power = VdotR * pow(VdotR, material->alpha);


    cuda_vec diffuse = NdotL * tocuda_vec(material->kd) * ray.w;
    cuda_vec specular = power * tocuda_vec(material->ks) * ray.w;

    cuda_vec finalColor = (diffuse + specular);
    return finalColor;
  }

   __device__ cuda_vec blinnShade(const gvt::render::data::primitives::Material* material,
                   const Ray &ray, const cuda_vec &N, const cuda_vec& wi) {

     float NdotL = fmaxf(0.f, (N * wi));

    cuda_vec H = normalize((wi - ray.direction));

    float NdotH = (H * N);
    float power = NdotH * pow(NdotH, material->alpha);

    cuda_vec diffuse = NdotL * tocuda_vec(material->kd) * ray.w;
    cuda_vec specular = power * tocuda_vec(material->ks) * ray.w;

    cuda_vec finalColor = (diffuse + specular);
    return finalColor;
  }


  __device__ bool gvt::render::data::cuda_primitives::Shade(gvt::render::data::primitives::Material *material,
                                                          const Ray &ray, const cuda_vec &sufaceNormal,
                                                          const Light *lightSource, const cuda_vec lightPosSample,
                                                          cuda_vec &r) {

  cuda_vec hitPoint = ray.origin + ray.direction * ray.t;
  cuda_vec wi = normalize(lightPosSample - hitPoint);
  float NdotL = fmaxf(0.f, (sufaceNormal * wi));
  cuda_vec Li = lightSource->contribution(hitPoint, lightPosSample);

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
