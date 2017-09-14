//
// Created by Joao Barbosa on 9/12/17.
//

#ifndef CONTEXT_VARIANT_DEF_H
#define CONTEXT_VARIANT_DEF_H

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <gvt/core/cntx/context.h>
#include <gvt/render/data/Primitives.h>

namespace cntx {

typedef details::variant<bool, int, float, double, unsigned, unsigned long, glm::vec3, std::string,
                         std::shared_ptr<glm::mat3>, std::shared_ptr<glm::mat4>,
                         std::shared_ptr<gvt::render::data::primitives::Mesh>,
                         std::shared_ptr<gvt::render::data::primitives::Box3D>, std::nullptr_t, identifier>
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

  for (int i = 0; i < 12; i++) pack<float>(m[i]);
}

unpack_function_signature(std::shared_ptr<glm::mat4>) {

  std::shared_ptr<glm::mat4> mptr = std::make_shared<glm::mat4>(1.f);
  float *m = glm::value_ptr(*mptr.get());

  for (int i = 0; i < 12; i++) m[i] = unpack<float>();

  return mptr;
}

pack_function_signature(std::shared_ptr<gvt::render::data::primitives::Mesh>) {

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

  std::shared_ptr<gvt::render::data::primitives::Mesh> mesh = std::make_shared<gvt::render::data::primitives::Mesh>();

  return mesh;
}

} // namespace mpi

} // namespace cntx

#endif // CONTEXT_VARIANT_DEF_H
