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
 * File:   gvt_mesh.cpp
 * Author: jbarbosa
 *
 * Created on April 20, 2014, 11:01 PM
 */
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/primitives/Mesh.h>

using namespace gvt::render::actor;
using namespace gvt::render::data;
using namespace gvt::render::data::primitives;
using namespace gvt::render::data::scene;

Mesh::Mesh(Material *mat) : mat(mat), haveNormals(false) {}

Mesh::Mesh(const Mesh &orig) {
  mat = orig.mat;
  vertices = orig.vertices;
  normals = orig.normals;
  faces = orig.faces;
  boundingBox = orig.boundingBox;
}

Mesh::~Mesh() { delete mat; }

void Mesh::addVertexNormalTexUV(glm::vec3 vertex, glm::vec3 normal, glm::vec3 texUV) {
  vertices.push_back(vertex);
  normals.push_back(normal);
  mapuv.push_back(texUV);
  boundingBox.expand(vertex);
}

void Mesh::addVertex(glm::vec3 vertex) {
  vertices.push_back(vertex);
  boundingBox.expand(vertex);
}

void Mesh::addNormal(glm::vec3 normal) { normals.push_back(normal); }

void Mesh::addTexUV(glm::vec3 texUV) { mapuv.push_back(texUV); }

void Mesh::setNormal(int which, glm::vec3 normal) {
  GVT_ASSERT((which > vertices.size()), "Setting normal outside the bounds");
  normals[which] = normal;
}

void Mesh::setTexUV(int which, glm::vec3 texUV) {
  GVT_ASSERT((which > vertices.size()), "Setting texture outside the bounds");
  this->mapuv[which] = texUV;
}

void Mesh::setVertex(int which, glm::vec3 vertex, glm::vec3 normal, glm::vec3 texUV) {
  GVT_ASSERT((which > vertices.size()), "Setting vertex outside the bounds");
  vertices[which] = vertex;
  boundingBox.expand(vertex);
  if (glm::length(normal)) normals[which] = normal;
  if (glm::length(texUV)) this->mapuv[which] = texUV;
}

void Mesh::setMaterial(Material *mat_) {
  this->mat = new Material();
  *(this->mat) = *mat_;
}

void Mesh::addFace(int v0, int v1, int v2) {
  GVT_ASSERT((v0 - 1 >= 0) && v0 - 1 < vertices.size(), "Vertex index 0 outside bounds : " << (v0 - 1));
  GVT_ASSERT((v1 - 1 >= 0) && v1 - 1 < vertices.size(), "Vertex index 1 outside bounds : " << (v1 - 1));
  GVT_ASSERT((v2 - 1 >= 0) && v2 - 1 < vertices.size(), "Vertex index 2 outside bounds : " << (v2 - 1));

  if (vertices[v0 - 1] == vertices[v1 - 1] || vertices[v1 - 1] == vertices[v2 - 1] ||
      vertices[v2 - 1] == vertices[v0 - 1])
    return;

  faces.push_back(Face(v0 - 1, v1 - 1, v2 - 1));
}

void Mesh::addFaceToNormals(Mesh::FaceToNormals face) { faces_to_normals.push_back(face); }

void Mesh::generateNormals() {
  if (haveNormals) return;
  normals.clear();
  face_normals.clear();
  faces_to_normals.clear();
  normals.resize(vertices.size());
  face_normals.resize(faces.size());
  faces_to_normals.resize(faces.size());
  for (int i = 0; i < normals.size(); ++i) normals[i] = glm::vec3(0.0f, 0.0f, 0.0f);

  for (int i = 0; i < faces.size(); ++i) {
    int I = faces[i].get<0>();
    int J = faces[i].get<1>();
    int K = faces[i].get<2>();
    glm::vec3 const &a = vertices[I];
    glm::vec3 const &b = vertices[J];
    glm::vec3 const &c = vertices[K];
    glm::vec3 u = b - a;
    glm::vec3 v = c - a;
    glm::vec3 normal;
    normal[0] = u[1] * v[2] - u[2] * v[1];
    normal[1] = u[2] * v[0] - u[0] * v[2];
    normal[2] = u[0] * v[1] - u[1] * v[0];
    normal = glm::normalize(normal);

    // glm::vec3 const &a = Triangle.Position[0];
    // glm::vec3 const &b = Triangle.Position[1];
    // glm::vec3 const &c = Triangle.Position[2];
    // Triangle.Normal = glm::normalize(glm::cross(c - a, b - a));

    face_normals[i] = normal;
    normals[I] += normal;
    normals[J] += normal;
    normals[K] += normal;
    faces_to_normals[i] = FaceToNormals(I, J, K);
  }
  for (int i = 0; i < normals.size(); ++i) normals[i] = glm::normalize(normals[i]);
  haveNormals = true;
}

// Color Mesh::shadeFace(const int face_id, const Ray &r, const glm::vec3 &normal, const Light *lsource) {
//  // XXX TODO: shadeFace returns constant color, fix?
//
//  if (!faces_to_materials.size()) return shade(r, normal, lsource, lsource->position);
//
//  Color c(0.5f, 0.5f, 0.5f);
//  Material *m = (faces_to_materials[face_id] ? faces_to_materials[face_id] : mat);
//  if (m) {
//    c = m->shade(r, normal, lsource, lsource->position);
//  }
//  return c;
//}
//
// Color Mesh::shadeFaceAreaLight(const int face_id, const Ray &r, const glm::vec3 &normal, const Light *lsource,
//                               const glm::vec3 areaLightPosition) {
//
//  if (!faces_to_materials.size()) return shade(r, normal, lsource, areaLightPosition);
//
//  Color c(0.5f, 0.5f, 0.5f);
//  Material *m = (faces_to_materials[face_id] ? faces_to_materials[face_id] : mat);
//  if (m) {
//    c = m->shade(r, normal, lsource, areaLightPosition);
//  }
//  return c;
//}
//
// Color Mesh::shade(const Ray &r, const glm::vec3 &normal, const Light *lsource, const glm::vec3 areaLightPosition) {
//  return mat->shade(r, normal, lsource, areaLightPosition);
//}

Box3D Mesh::computeBoundingBox() {

  Box3D box;
  for (int i = 0; i < vertices.size(); i++) {

    glm::vec3 p = vertices[i];

    box.expand(p);
  }

  this->boundingBox = box;

  return box;
}

void Mesh::writeobj(std::string filename) {

  std::ofstream file;
  file.open(filename);
  {
    file << "#vertices " << vertices.size() << std::endl;
    for (auto &v : vertices) file << "v " << v[0] << " " << v[1] << " " << v[2] << std::endl;

    file << "#vertices normal " << normals.size() << std::endl;
    for (auto &vn : normals) file << "vn " << vn[0] << " " << vn[1] << " " << vn[2] << std::endl;

    file << "#vertices " << faces.size() << std::endl;
    for (auto &f : faces) file << "f " << f.get<0>() + 1 << " " << f.get<1>() + 1 << " " << f.get<2>() + 1 << std::endl;
    file.close();
  }
}
