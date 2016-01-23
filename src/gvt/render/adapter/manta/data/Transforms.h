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
/*
 * File:   GVT_MANTA.h
 * Author: jbarbosa
 *
 * Created on March 30, 2014, 3:53 PM
 */

#ifndef GVT_RENDER_ADAPTER_MANTA_DATA_TRANSFORMS_H
#define GVT_RENDER_ADAPTER_MANTA_DATA_TRANSFORMS_H

#include <gvt/core/data/Transform.h>

#include <gvt/core/Math.h>
#include <gvt/render/actor/Ray.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/Light.h>

// begin Manta includes
#include <Core/Exceptions/Exception.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/vecdefs.h>
#include <Interface/Context.h>
#include <Interface/LightSet.h>
#include <Interface/MantaInterface.h>
#include <Interface/Object.h>
#include <Interface/Scene.h>
#include <Model/Groups/DynBVH.h>
#include <Model/Groups/Mesh.h>
#include <Model/Lights/PointLight.h>
#include <Model/Materials/Lambertian.h>
#include <Model/Materials/Phong.h>
#include <Model/Primitives/KenslerShirleyTriangle.h>
#include <Model/Readers/PlyReader.h>
// end Manta includes

#include <vector>

namespace gvt {
namespace render {
namespace adapter {
namespace manta {
namespace data {

GVT_TRANSFORM_TEMPLATE // see gvt/core/data/Transform.h

    // clang-format off
    /// return a Manta-compliant 4-float vector
    template <>
struct transform_impl<gvt::core::math::Vector4f, Manta::Vec4f> {

  static inline Manta::Vec4f transform(const gvt::core::math::Vector4f &r) {
    return Manta::Vec4f(r[0], r[1], r[2], r[3]);
  }
};
// clang-format on

/// return a Manta-compliant 4-float vector
template <> struct transform_impl<gvt::core::math::Point4f, Manta::Vec4f> {

  static inline Manta::Vec4f transform(const gvt::core::math::Point4f &r) {
    return Manta::Vec4f(r[0], r[1], r[2], r[3]);
  }
};

/// return a Manta-compliant vector
template <> struct transform_impl<gvt::core::math::Point4f, Manta::Vector> {

  static inline Manta::Vector transform(const gvt::core::math::Point4f &r) { return Manta::Vector(r[0], r[1], r[2]); }
};

/// return a Manta-compliant vector
template <> struct transform_impl<gvt::core::math::Vector4f, Manta::Vector> {

  static inline Manta::Vector transform(const gvt::core::math::Vector4f &r) { return Manta::Vector(r[0], r[1], r[2]); }
};

/// return a GraviT-compliant Point
template <> struct transform_impl<Manta::Vector, gvt::core::math::Point4f> {

  static inline gvt::core::math::Point4f transform(const Manta::Vector &r) {
    return gvt::core::math::Point4f(r[0], r[1], r[2], 1.f);
  }
};

/// return a GraviT-compliant Vector
template <> struct transform_impl<Manta::Vector, gvt::core::math::Vector4f> {

  static inline gvt::core::math::Vector4f transform(const Manta::Vector &r) {
    return gvt::core::math::Point4f(r[0], r[1], r[2], 0.f);
    // return Manta::Vector(r[0], r[1], r[2]);
  }
};

/// return a Manta-compliant ray
template <> struct transform_impl<gvt::render::actor::Ray, Manta::Ray> {
  static inline Manta::Ray transform(const gvt::render::actor::Ray &r) {
    Manta::Ray ray;
    const Manta::Vector orig =
        gvt::render::adapter::manta::data::transform<gvt::core::math::Point4f, Manta::Vector>(r.origin);
    const Manta::Vector dir =
        gvt::render::adapter::manta::data::transform<gvt::core::math::Vector4f, Manta::Vector>(r.direction);
    ray.set(orig, dir);
    return ray;
  }
};

/// return a GraviT-compliant ray
template <> struct transform_impl<Manta::Ray, gvt::render::actor::Ray> {
  static inline gvt::render::actor::Ray transform(const Manta::Ray &r) {
    return gvt::render::actor::Ray(
        gvt::render::adapter::manta::data::transform<Manta::Vector, gvt::core::math::Point4f>(r.origin()),
        gvt::render::adapter::manta::data::transform<Manta::Vector, gvt::core::math::Vector4f>(r.direction()));
  }
};

/// return a Manta-compliant point light
template <> struct transform_impl<Manta::PointLight *, gvt::render::data::scene::PointLight *> {

  static inline Manta::PointLight *transform(const gvt::render::data::scene::PointLight *ls) {
    // lights->add(new Manta::PointLight(Manta::Vector(0, -5, 8), Manta::Color(Manta::RGBColor(1, 1, 1))));

    return new Manta::PointLight(Manta::Vector(ls->position[0], ls->position[1], ls->position[2]),
                                 Manta::Color(Manta::RGBColor(ls->color[0], ls->color[1], ls->color[2])));
    // return gvt::core::math::Point4f(r[0], r[1], r[2], 0.f);
    // return Manta::Vector(r[0], r[1], r[2]);
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
//                    Manta::RayPacket* mRays = new Manta::RayPacket(*rpData, Manta::RayPacket::UnknownShape, 0, psize,
//                    0, Manta::RayPacket::NormalizedDirections);
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

/// return a GraviT-compliant Mesh
template <> struct transform_impl<Manta::Mesh *, gvt::render::data::primitives::Mesh *> {

  static inline gvt::render::data::primitives::Mesh *transform(Manta::Mesh *mesh) {
    gvt::render::data::primitives::Mesh *gvtmesh = new gvt::render::data::primitives::Mesh(NULL);

    int count_vertex = 0;

    for (int i = 0; i < mesh->vertices.size(); i++) {
      gvt::core::math::Point4f vertex =
          gvt::render::adapter::manta::data::transform<Manta::Vector, gvt::core::math::Point4f>(mesh->vertices[i]);
      gvtmesh->addVertex(vertex);
    }

    for (int i = 0; i < mesh->vertex_indices.size(); i += 3) {
      gvtmesh->addFace(mesh->vertex_indices[i], mesh->vertex_indices[i + 1], mesh->vertex_indices[i + 2]);
    }

    return gvtmesh;
  }
};

/// return a Manta-compliant Mesh
template <> struct transform_impl<gvt::render::data::primitives::Mesh *, Manta::Mesh *> {

  static inline Manta::Mesh *transform(gvt::render::data::primitives::Mesh *mesh) {
    Manta::Mesh *m = new Manta::Mesh();
    m->materials.push_back(new Manta::Lambertian(Manta::Color(Manta::RGBColor(0.f, 0.f, 0.f))));

    for (int i = 0; i < mesh->vertices.size(); i++) {
      Manta::Vector v0 =
          gvt::render::adapter::manta::data::transform<gvt::core::math::Vector4f, Manta::Vector>(mesh->vertices[i]);
      m->vertices.push_back(v0);
    }
    for (int i = 0; i < mesh->normals.size(); i++) {
      Manta::Vector v0 =
          gvt::render::adapter::manta::data::transform<gvt::core::math::Vector4f, Manta::Vector>(mesh->normals[i]);
      m->vertexNormals.push_back(v0);
    }
    for (int i = 0; i < mesh->faces.size(); i++) {
      gvt::render::data::primitives::Mesh::Face f = mesh->faces[i];
      m->texture_indices.push_back(Manta::Mesh::kNoTextureIndex);
      m->texture_indices.push_back(Manta::Mesh::kNoTextureIndex);
      m->texture_indices.push_back(Manta::Mesh::kNoTextureIndex);
      m->vertex_indices.push_back(boost::get<0>(f));
      m->vertex_indices.push_back(boost::get<1>(f));
      m->vertex_indices.push_back(boost::get<2>(f));
      m->normal_indices.push_back(boost::get<0>(f));
      m->normal_indices.push_back(boost::get<1>(f));
      m->normal_indices.push_back(boost::get<2>(f));
      m->face_material.push_back(0);
      m->addTriangle(new Manta::KenslerShirleyTriangle());
    }
    return m;
  }
};
}
}
}
}
}

#endif /* GVT_RENDER_ADAPTER_MANTA_DATA_TRANSFORMS_H */
