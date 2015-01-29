/* 
 * File:   GVT_MANTA.h
 * Author: jbarbosa
 *
 * Created on March 30, 2014, 3:53 PM
 */

#ifndef GVT_MANTA_H
#define	GVT_MANTA_H

#include <Model/Materials/Lambertian.h>

#include <GVT/Math/GVTMath.h>
#include <GVT/Data/transform.h>
#include <GVT/Data/primitives.h>
#include <GVT/Data/primitives/gvt_ray.h>

#include <Core/Geometry/Ray.h>
#include <vector>

#include <Model/Primitives/KenslerShirleyTriangle.h>
#include <Interface/MantaInterface.h>
#include <Interface/Scene.h>
#include <Interface/Object.h>
#include <Interface/Context.h>
#include <Core/Geometry/BBox.h>
#include <Core/Exceptions/Exception.h>
#include <Core/Exceptions/InternalError.h>
#include <Model/Groups/DynBVH.h>
#include <Model/Groups/Mesh.h>
#include <Model/Materials/Phong.h>
#include <Model/Readers/PlyReader.h>
#include <Interface/LightSet.h>
#include <Model/Lights/PointLight.h>

#include <GVT/Data/primitives.h>

namespace GVT {
    namespace Data {

        template<>
        struct transform_impl<GVT::Math::Vector4f, Manta::Vec4f > {

            static inline Manta::Vec4f transform(const GVT::Math::Vector4f& r) {
                return Manta::Vec4f(r[0], r[1], r[2], r[3]);
            }
        };

        template<>
        struct transform_impl<GVT::Math::Point4f, Manta::Vec4f > {

            static inline Manta::Vec4f transform(const GVT::Math::Point4f& r) {
                return Manta::Vec4f(r[0], r[1], r[2], r[3]);
            }
        };

        template<>
        struct transform_impl<GVT::Math::Point4f, Manta::Vector > {

            static inline Manta::Vector transform(const GVT::Math::Point4f& r) {
                return Manta::Vector(r[0], r[1], r[2]);
            }
        };

        template<>
        struct transform_impl<GVT::Math::Vector4f, Manta::Vector> {

            static inline Manta::Vector transform(const GVT::Math::Vector4f& r) {
                return Manta::Vector(r[0], r[1], r[2]);
            }
        };

        template<>
        struct transform_impl<Manta::Vector, GVT::Math::Point4f > {

            static inline GVT::Math::Point4f transform(const Manta::Vector& r) {
                return GVT::Math::Point4f(r[0], r[1], r[2], 1.f);
            }
        };

        template<>
        struct transform_impl<Manta::Vector, GVT::Math::Vector4f > {

            static inline GVT::Math::Vector4f transform(const Manta::Vector& r) {
                return GVT::Math::Point4f(r[0], r[1], r[2], 0.f);
                //return Manta::Vector(r[0], r[1], r[2]);
            }
        };

        template<>
        struct transform_impl<GVT::Data::ray, Manta::Ray> {
            static inline Manta::Ray transform(const GVT::Data::ray& r) {
                Manta::Ray ray;
                const Manta::Vector orig = GVT::Data::transform<GVT::Math::Point4f, Manta::Vector>(r.origin);
                const Manta::Vector dir = GVT::Data::transform<GVT::Math::Vector4f, Manta::Vector>(r.direction);
                ray.set(orig, dir);
                return ray;
            }
        };

        template<>
        struct transform_impl<Manta::Ray, GVT::Data::ray> {
            static inline GVT::Data::ray transform(const Manta::Ray& r) {
                return GVT::Data::ray(
                        GVT::Data::transform<Manta::Vector, GVT::Math::Point4f>(r.origin()),
                        GVT::Data::transform<Manta::Vector, GVT::Math::Vector4f>(r.direction()));

            }
        };
        
        
        template<>
        struct transform_impl<Manta::PointLight*, GVT::Data::PointLightSource* > {

            static inline Manta::PointLight* transform(const GVT::Data::PointLightSource* ls) {
                // lights->add(new Manta::PointLight(Manta::Vector(0, -5, 8), Manta::Color(Manta::RGBColor(1, 1, 1))));

                return new Manta::PointLight(
                        Manta::Vector(ls->position[0],ls->position[1],ls->position[2]),
                        Manta::Color(Manta::RGBColor(ls->color[0],ls->color[1],ls->color[2]))
                        );
                //return GVT::Math::Point4f(r[0], r[1], r[2], 0.f);
                //return Manta::Vector(r[0], r[1], r[2]);
            }
        };

//        template<size_t LENGTH>
//        struct transform_impl<GVT::Data::RayVector, std::vector<Manta::RayPacket*>, LENGTH> {
//
//            static inline std::vector<Manta::RayPacket*> transform(const GVT::Data::RayVector& rr) {
//                GVT_DEBUG(DBG_ALWAYS, "Called the right transform");
//                std::vector<Manta::RayPacket*> ret;
//                size_t current_ray = 0;
//                for (; current_ray < rr.size();) {
//                    size_t psize = std::min(LENGTH, (rr.size() - current_ray));
//                    Manta::RayPacketData* rpData = new Manta::RayPacketData();
//                    Manta::RayPacket* mRays = new Manta::RayPacket(*rpData, Manta::RayPacket::UnknownShape, 0, psize, 0, Manta::RayPacket::NormalizedDirections);
//                    
//                    for (int i = 0; i < psize; i++) {
//                        mRays->setRay(i, GVT::Data::transform<GVT::Data::ray, Manta::Ray>(rr[current_ray + i]));
//                    }
//                    
//                    ret.push_back(mRays);
//                    current_ray += psize;
//                }
//                return ret;
//            }
//        };

        template<>
        struct transform_impl<Manta::Mesh*, GVT::Data::Mesh*> {

            static inline GVT::Data::Mesh* transform(Manta::Mesh* mesh) {
                GVT::Data::Mesh* gvtmesh = new GVT::Data::Mesh(NULL);

                int count_vertex = 0;

                for (int i = 0; i < mesh->vertices.size(); i++) {
                    GVT::Math::Point4f vertex = GVT::Data::transform<Manta::Vector, GVT::Math::Point4f>(mesh->vertices[i]);
                    gvtmesh->vertices.push_back(vertex);
                    gvtmesh->boundingBox.expand(vertex);
                }

                for (int i = 0; i < mesh->vertex_indices.size(); i += 3) {
                    gvtmesh->faces.push_back(GVT::Data::Mesh::face(mesh->vertex_indices[i], mesh->vertex_indices[i + 1], mesh->vertex_indices[i + 2]));
                }

                return gvtmesh;
            }
        };

        template<>
        struct transform_impl<GVT::Data::Mesh*, Manta::Mesh*> {

            static inline Manta::Mesh* transform(GVT::Data::Mesh* mesh) {
                Manta::Mesh* m = new Manta::Mesh();
                m->materials.push_back(new Manta::Lambertian(Manta::Color(Manta::RGBColor(0.f, 0.f, 0.f))));

                for(int i=0; i < mesh->vertices.size(); i++) {
                    Manta::Vector v0 = GVT::Data::transform<GVT::Math::Vector4f, Manta::Vector>(mesh->vertices[i]);
                    m->vertices.push_back(v0);
                }
                for (int i = 0; i < mesh->faces.size(); i++) {
                    GVT::Data::Mesh::face f = mesh->faces[i];
                    m->texture_indices.push_back(Manta::Mesh::kNoTextureIndex);
                    m->texture_indices.push_back(Manta::Mesh::kNoTextureIndex);
                    m->texture_indices.push_back(Manta::Mesh::kNoTextureIndex);
                    m->vertex_indices.push_back(boost::get<0>(f));
                    m->vertex_indices.push_back(boost::get<1>(f));
                    m->vertex_indices.push_back(boost::get<2>(f));
                    m->face_material.push_back(0);
                    m->addTriangle(new Manta::KenslerShirleyTriangle());
                }
                return m;
            }
        };


    }
}

#endif	/* GVT_MANTA_H */

