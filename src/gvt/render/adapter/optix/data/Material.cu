

#include "cutil_math.h"
#include "Material.cuh"


using namespace gvt::render::data::cuda_primitives;





   __device__  float4 BaseMaterial::CosWeightedRandomHemisphereDirection2(float4 n) {

    float Xi1 = cudaRand();
    float Xi2 = cudaRand();

    float theta = acos(sqrt(1.0 - Xi1));
    float phi = 2.0 * 3.1415926535897932384626433832795 * Xi2;

    float xs = sinf(theta) * cosf(phi);
    float ys = cosf(theta);
    float zs = sinf(theta) * sinf(phi);

    float3 y = make_float3(n);
    float3 h = y;
    if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z))
      h.x = 1.0;
    else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z))
      h.y = 1.0;
    else
      h.z = 1.0;

    float3 x = cross(h,y);//(h ^ y);
    float3 z = cross(x, y);

    float4 direction = make_float4(x * xs + y * ys + z * zs);
    return normalize(direction);
  }



 /* Material::Material() {}s

  Material::Material(const Material &orig) {}

  Material::~Material() {}
*/
/*
  float4 BaseMaterial::shade(const Ray &ray, const float4 &sufaceNormal, const Light *lightSource) {
	  return make_float4(0.f);
  }
*/

  /*RayVector Material::ao(const Ray &ray, const float4 &sufaceNormal, float samples) { return RayVector(); }

  RayVector Material::secondary(const Ray &ray, const float4 &sufaceNormal, float samples) { return RayVector(); }

  Lambert::Lambert(const float4 &kd) : Material(), kd(kd) {}

  Lambert::Lambert(const Lambert &orig) : Material(orig), kd(orig.kd) {}

  Lambert::~Lambert() {}
*/
   __device__ float4 Lambert::shade( const Ray &ray, const float4 &N, const Light *lightSource) {


    float4 hitPoint = ray.origin + ray.direction * ray.t;
    float4 L = normalize(lightSource->light.position - hitPoint);
    float NdotL = fmaxf(0.f, fabs(N * L));
    Color lightSourceContrib = lightSource->contribution(hitPoint);

    Color diffuse = prod(lightSourceContrib, kd) * (NdotL * ray.w);


    return diffuse;
  }
/*
  RayVector Lambert::ao(const Ray &ray, const float4 &sufaceNormal, float samples) { return RayVector(); }

  RayVector Lambert::secundary(const Ray &ray, const float4 &sufaceNormal, float samples) { return RayVector(); }

  Phong::Phong(const float4 &kd, const float4 &ks, const float &alpha) : Material(), kd(kd), ks(ks), alpha(alpha) {}

  Phong::Phong(const Phong &orig) : Material(orig), kd(orig.kd), ks(orig.ks), alpha(orig.alpha) {}

  Phong::~Phong() {}

  */

   __device__ float4 Phong::shade(const Ray &ray, const float4 &N, const Light *lightSource) {


   float4 hitPoint = ray.origin + (ray.direction * ray.t);
   float4 L =normalize(lightSource->light.position - hitPoint);

    float NdotL =fmaxf(0.f, (N * L));
    float4 R = ((N * 2.f) * NdotL) - L;
    float4 invDir = make_float4(-ray.direction.x, -ray.direction.y, -ray.direction.z, -ray.direction.w);
    float VdotR = fmaxf(0.f, (R * invDir));
    float power = VdotR * pow(VdotR, alpha);

    float4 lightSourceContrib = lightSource->contribution(hitPoint); //  distance;

    Color finalColor = prod(lightSourceContrib , kd) * (NdotL * ray.w);
    finalColor += prod(lightSourceContrib , ks) * (power * ray.w);
    return finalColor;


    return finalColor;
  }

  /*

  RayVector Phong::ao(const Ray &ray, const float4 &sufaceNormal, float samples) { return RayVector(); }

  RayVector Phong::secundary(const Ray &ray, const float4 &sufaceNormal, float samples) { return RayVector(); }

  BlinnPhong::BlinnPhong(const float4 &kd, const float4 &ks, const float &alpha)
      : Material(), kd(kd), ks(ks), alpha(alpha) {}

  BlinnPhong::BlinnPhong(const BlinnPhong &orig) : Material(orig), kd(orig.kd), ks(orig.ks), alpha(orig.alpha) {}

  BlinnPhong::~BlinnPhong() {}
*/
   __device__ float4 BlinnPhong::shade(const Ray &ray, const float4 &N, const Light *lightSource) {
//    float4 hitPoint = (float4)ray.origin + (ray.direction * ray.t);
//    float4 L = (float4)lightSource->light.position - hitPoint;
//    L = normalize(L);
//    float NdotL = fmaxf(0.f, (N * L));
//
//    float4 H = normalize((L - ray.direction));
//
//    float NdotH = (H * N);
//    float power = NdotH * std::pow(NdotH, alpha);
//
//    float4 lightSourceContrib = lightSource->contribution(ray);
//
//    Color diffuse = prod((lightSourceContrib * NdotL), kd) * ray.w;
//    Color specular = prod((lightSourceContrib * power), ks) * ray.w;
//
//    Color finalColor = (diffuse + specular);
//    return finalColor;

	   float4 hitPoint = ray.origin + (ray.direction * ray.t);
	   float4 L = normalize(lightSource->light.position - hitPoint);
	   float NdotL = fmaxf(0.f, (N* L));

	   float4 H = normalize(L - ray.direction);

	   float NdotH = fmaxf(0.f, (H * N));
	   float power = NdotH * pow(NdotH, alpha);

	   float4 lightSourceContrib = lightSource->contribution(hitPoint);

	   Color diffuse = prod(lightSourceContrib , kd) * (NdotL * ray.w);
	   Color specular = prod(lightSourceContrib , ks) * (power * ray.w);

	   Color finalColor = (diffuse + specular);
	   return finalColor;
  }
/*
  RayVector BlinnPhong::ao(const Ray &ray, const float4 &sufaceNormal, float samples) { return RayVector(); }

  RayVector BlinnPhong::secundary(const Ray &ray, const float4 &sufaceNormal, float samples) { return RayVector(); }*/
