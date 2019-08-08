/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray tracing
   tacc.github.io/GraviT

   Copyright 2013-2017 Texas Advanced Computing Center, The University of Texas at Austin
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

//
// Created by Joao Barbosa on 9/12/17.
//

#ifndef GVT_CONTEXT_VARIANT_DEF_H
#define GVT_CONTEXT_VARIANT_DEF_H

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <gvt/core/cntx/context.h>
#include <gvt/render/data/Primitives.h>

namespace cntx {

typedef details::variant<bool, int, float, double, unsigned, unsigned long, glm::vec3, std::string,

                         std::shared_ptr<glm::mat3>, std::shared_ptr<glm::mat4>,
                         std::shared_ptr<gvt::render::data::primitives::Box3D>,
                         std::shared_ptr<gvt::render::data::primitives::Mesh>,
#ifdef GVT_BUILD_VOLUME
                         std::shared_ptr<gvt::render::data::primitives::Volume>,
#endif                         
                         std::shared_ptr<std::vector<int> >,
                         identifier, std::nullptr_t>
    Variant;

namespace mpi {

pack_function_signature(glm::vec3) {
  pack<float>(v[0]);
  pack<float>(v[1]);
  pack<float>(v[2]);
}

unpack_function_signature(glm::vec3) {
  glm::vec3 v;
  v[0] = unpack<float>();
  v[1] = unpack<float>();
  v[2] = unpack<float>();
  return v;
}

pack_function_signature(std::shared_ptr<glm::mat3>) {

  float *m = glm::value_ptr(*v.get());

  for (int i = 0; i < 9; i++) pack<float>(m[i]);
}

unpack_function_signature(std::shared_ptr<glm::mat3>) {

  std::shared_ptr<glm::mat3> mptr = std::make_shared<glm::mat3>(1.f);
  float *m = glm::value_ptr(*mptr.get());

  for (int i = 0; i < 9; i++) m[i] = unpack<float>();

  return mptr;
}

pack_function_signature(std::shared_ptr<glm::mat4>) {

  float *m = glm::value_ptr(*v.get());
  for (int i = 0; i < 16; i++) {
    pack<float>(m[i]);
  }
}

unpack_function_signature(std::shared_ptr<glm::mat4>) {

  std::shared_ptr<glm::mat4> mptr = std::make_shared<glm::mat4>(1.f);
  float *m = glm::value_ptr(*mptr.get());
  for (int i = 0; i < 16; i++) {
    m[i] = unpack<float>();
  }

  return mptr;
}

pack_function_signature(std::shared_ptr<gvt::render::data::primitives::Box3D>) {
  pack<glm::vec3>(v->bounds_min);
  pack<glm::vec3>(v->bounds_max);
}

unpack_function_signature(std::shared_ptr<gvt::render::data::primitives::Box3D>) {
  gvt::render::data::primitives::Box3D bb;
  bb.bounds_min = unpack<glm::vec3>();
  bb.bounds_max = unpack<glm::vec3>();
  return std::make_shared<gvt::render::data::primitives::Box3D>(bb);
}

pack_function_signature(std::shared_ptr<std::vector<int> >) {

  pack<size_t>(v->size());
  for (int i = 0; i < v->size(); i++) {
    pack<int>((*v.get())[i]);
  }
}

unpack_function_signature(std::shared_ptr<std::vector<int> >) {

  size_t size = unpack<size_t>();
  std::shared_ptr<std::vector<int> > v = std::make_shared<std::vector<int> >();
  ;
  for (int i = 0; i < size; i++) {
    v->push_back(unpack<int>());
  }
  return v;
}

pack_function_signature(std::shared_ptr<gvt::render::data::primitives::Mesh>) {

  pack<std::nullptr_t>(nullptr);

#if 0
  pack<bool>(v == nullptr);
  if (v != nullptr) {

    gvt::render::data::primitives::Mesh &mesh = *v.get();

    pack<bool>(mesh.haveNormals);
    pack<glm::vec3>(mesh.boundingBox.bounds_min);
    pack<glm::vec3>(mesh.boundingBox.bounds_max);
    pack<size_t>(mesh.vertices.size() * sizeof(glm::vec3));
    pack<size_t>(mesh.faces.size());
    pack<size_t>(mesh.normals.size());
    pack<size_t>(mesh.faces_to_materials.size());
    pack<size_t>(mesh.faces_to_normals.size());
    pack<size_t>(mesh.mapuv.size());
    //      pack<size_t>(mesh.mat.size());

    pack<glm::vec3>(&mesh.vertices[0], mesh.vertices.size() * sizeof(glm::vec3));
  }
#endif
  //    pack<float>(v[0]);
  //    pack<float>(v[1]);
  //    pack<float>(v[2]);
}

unpack_function_signature(std::shared_ptr<gvt::render::data::primitives::Mesh>) {
  //    glm::vec3 v;
  //    v[0] = unpack<float>();
  //    v[1] = unpack<float>();
  //    v[2] = unpack<float>();
  //    return v;
#if 0
  std::shared_ptr<gvt::render::data::primitives::Mesh> mesh = std::make_shared<gvt::render::data::primitives::Mesh>();
#endif

  return nullptr;
}

#ifdef GVT_BUILD_VOLUME
pack_function_signature(std::shared_ptr<gvt::render::data::primitives::Volume>) {
  pack<std::nullptr_t>(nullptr);
}

unpack_function_signature(std::shared_ptr<gvt::render::data::primitives::Volume>) {
  return nullptr;
}
#endif

} // namespace mpi

namespace details {

inline std::ostream &operator<<(std::ostream &os, const std::shared_ptr<gvt::render::data::primitives::Box3D> &other) {
  return (os << other->bounds_min << " x " << other->bounds_max);
}

} // namespace details

} // namespace cntx

#endif // GVT_CONTEXT_VARIANT_DEF_H
