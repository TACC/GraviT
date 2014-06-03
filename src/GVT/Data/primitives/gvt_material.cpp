/* 
 * File:   gvt_material.cpp
 * Author: jbarbosa
 * 
 * Created on April 18, 2014, 3:07 PM
 */
#include <cmath>
#include <GVT/Data/derived_types.h>
#include "gvt_material.h"
namespace GVT {
    namespace Data {

        Material::Material() {
        }

        Material::Material(const Material& orig) {
        }

        Material::~Material() {
        }

        GVT::Math::Vector4f Material::shade(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, const GVT::Data::LightSource* lightSource) {
            return GVT::Math::Vector4f();
        }

        GVT::Data::RayVector Material::ao(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples) {
            return GVT::Data::RayVector();
        }

        GVT::Data::RayVector Material::secundary(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples) {
            return GVT::Data::RayVector();
        }

        Lambert::Lambert(const GVT::Math::Vector4f& kd) : Material(), kd(kd) {
        }

        Lambert::Lambert(const Lambert& orig) : Material(orig), kd(orig.kd) {
        }

        Lambert::~Lambert() {
        }

        GVT::Math::Vector4f Lambert::shade(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& N, const GVT::Data::LightSource* lightSource) {
            
            GVT::Math::Point4f L = ray.direction;
            L = L.normalize();
            float NdotL = std::max(0.f, N*L);
            GVT::Data::Color lightSourceContrib = lightSource->contribution(ray);
            GVT::Data::Color diffuse = GVT::Math::prod(lightSourceContrib, kd * NdotL) * ray.w;
            return diffuse;
        }

        GVT::Data::RayVector Lambert::ao(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples) {
            return GVT::Data::RayVector();
        }

        GVT::Data::RayVector Lambert::secundary(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples) {
            return GVT::Data::RayVector();
        }

        Phong::Phong(const GVT::Math::Vector4f& kd, const GVT::Math::Vector4f& ks, const float& alpha) : Material(),
        kd(kd), ks(ks), alpha(alpha) {
        }

        Phong::Phong(const Phong& orig) : Material(orig), kd(orig.kd), ks(orig.ks), alpha(orig.alpha) {
        }

        Phong::~Phong() {
        }

        GVT::Math::Vector4f Phong::shade(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& N, const GVT::Data::LightSource* lightSource) {
            GVT::Math::Vector4f hitPoint = (GVT::Math::Vector4f)ray.origin + (ray.direction * ray.t);
            GVT::Math::Vector4f L = (GVT::Math::Vector4f)lightSource->position - hitPoint;
            
            L = L.normalize();
            float NdotL = std::max(0.f, (N * L));
            GVT::Math::Vector4f R = ((N * 2.f) * NdotL) - L;
            float VdotR = std::max(0.f, (R * (-ray.direction)));
            float power = VdotR * std::pow(VdotR, alpha);

            GVT::Math::Vector4f lightSourceContrib = lightSource->contribution(ray); //  distance;

            GVT::Data::Color diffuse = GVT::Math::prod((lightSourceContrib *NdotL), kd) * ray.w;
            GVT::Data::Color specular = GVT::Math::prod((lightSourceContrib * power), ks) * ray.w;

            GVT::Data::Color finalColor = (diffuse + specular);
            return finalColor;
        }

        GVT::Data::RayVector Phong::ao(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples) {
            return GVT::Data::RayVector();
        }

        GVT::Data::RayVector Phong::secundary(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples) {
            return GVT::Data::RayVector();
        }

        BlinnPhong::BlinnPhong(const GVT::Math::Vector4f& kd, const GVT::Math::Vector4f& ks, const float& alpha) : Material(),
        kd(kd), ks(ks), alpha(alpha) {
        }

        BlinnPhong::BlinnPhong(const BlinnPhong& orig) : Material(orig), kd(orig.kd), ks(orig.ks), alpha(orig.alpha) {
        }

        BlinnPhong::~BlinnPhong() {
        }

        GVT::Math::Vector4f BlinnPhong::shade(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& N, const GVT::Data::LightSource* lightSource) {
            GVT::Math::Vector4f hitPoint = (GVT::Math::Vector4f)ray.origin + (ray.direction * ray.t);
            GVT::Math::Vector4f L = (GVT::Math::Vector4f)lightSource->position - hitPoint;
            L = L.normalize();
            float NdotL = std::max(0.f, (N * L));

            GVT::Math::Vector4f H = (L - ray.direction).normalize();

            float NdotH = (H * N);
            float power = NdotH * std::pow(NdotH, alpha);

            GVT::Math::Vector4f lightSourceContrib = lightSource->contribution(ray);

            GVT::Data::Color diffuse = GVT::Math::prod((lightSourceContrib * NdotL) , kd) * ray.w;
            GVT::Data::Color specular = GVT::Math::prod((lightSourceContrib * power) ,  ks) * ray.w;

            GVT::Data::Color finalColor = (diffuse + specular);
            return finalColor;
        }

        GVT::Data::RayVector BlinnPhong::ao(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples) {
            return GVT::Data::RayVector();
        }

        GVT::Data::RayVector BlinnPhong::secundary(const GVT::Data::ray&  ray, const GVT::Math::Vector4f& sufaceNormal, float samples) {
            return GVT::Data::RayVector();
        }

    }
}