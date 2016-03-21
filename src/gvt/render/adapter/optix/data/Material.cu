

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

    float4 hitPoint = ray.origin + ray.direction * ray.t;
    float4 L = normalize(lightSource->light.position - hitPoint);
    float NdotL = fmaxf(0.f, fabs(N * L));
    float4 lightSourceContrib = lightSource->contribution(hitPoint);
    float4 diffuse = prod(lightSourceContrib, material->kd) * (NdotL * ray.w);
    return diffuse;
  }


   __device__ float4 phongShade(const gvt::render::data::primitives::CUDAPhong* material,
		   const Ray &ray, const float4 &N, const Light *lightSource) {


	float4 hitPoint = (float4)ray.origin + (ray.direction * ray.t);
    float4 L = (float4)lightSource->light.position - hitPoint;

    L = normalize(L);
    float NdotL = fmaxf(0.f, (N * L));
    float4 R = ((N * 2.f) * NdotL) - L;
    float VdotR = max(0.f, (R * (-1*ray.direction)));
    float power = VdotR * pow(VdotR, material->alpha);

    float4 lightSourceContrib = lightSource->contribution(hitPoint); //  distance;

    float4 diffuse = prod((lightSourceContrib * NdotL), material->kd) * ray.w;
    float4 specular = prod((lightSourceContrib * power), material->ks) * ray.w;

    float4 finalColor = (diffuse + specular);
    return finalColor;
  }

   __device__ float4 blinnShade(const gvt::render::data::primitives::CUDABlinnPhong* material,
		   const Ray &ray, const float4 &N, const Light *lightSource) {
    float4 hitPoint = (float4)ray.origin + (ray.direction * ray.t);
    float4 L = (float4)lightSource->light.position - hitPoint;
    L = normalize(L);
    float NdotL = fmaxf(0.f, (N * L));

    float4 H = normalize((L - ray.direction));

    float NdotH = (H * N);
    float power = NdotH * pow(NdotH, material->alpha);

    float4 lightSourceContrib = lightSource->contribution(hitPoint);

    float4 diffuse = prod((lightSourceContrib * NdotL), material->kd) * ray.w;
    float4 specular = prod((lightSourceContrib * power), material->ks) * ray.w;

    float4 finalColor = (diffuse + specular);
    return finalColor;
  }


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
                   case CUDA_PHONG:
                           r = phongShade((CUDAPhong*)material,
                        		   ray, sufaceNormal, lightSource);
                           break;
                   case CUDA_BLINN:
                           r = blinnShade((CUDABlinnPhong*)material,
                        		   ray, sufaceNormal, lightSource);
                           break;
                   default:
                	   printf("Unknown material type\n");
                           break;
                   }
                 return r;


   };
