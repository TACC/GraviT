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

using namespace gvt::core::math;
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

void Mesh::addVertexNormalTexUV(Point4f vertex, Vector4f normal, Point4f texUV) {
  vertices.push_back(vertex);
  normals.push_back(normal);
  mapuv.push_back(texUV);
  boundingBox.expand(vertex);
}

void Mesh::addVertex(Point4f vertex) {
  vertices.push_back(vertex);
  boundingBox.expand(vertex);
}

void Mesh::addNormal(Vector4f normal) { normals.push_back(normal); }

void Mesh::addTexUV(Point4f texUV) { mapuv.push_back(texUV); }

void Mesh::setNormal(int which, Vector4f normal) {
  GVT_ASSERT((which > vertices.size()), "Setting normal outside the bounds");
  normals[which] = normal;
}

void Mesh::setTexUV(int which, Point4f texUV) {
  GVT_ASSERT((which > vertices.size()), "Setting texture outside the bounds");
  this->mapuv[which] = texUV;
}

void Mesh::setVertex(int which, Point4f vertex, Vector4f normal, Point4f texUV) {
  GVT_ASSERT((which > vertices.size()), "Setting vertex outside the bounds");
  vertices[which] = vertex;
  boundingBox.expand(vertex);
  if (normal.length2()) normals[which] = normal;
  if (texUV.length2()) this->mapuv[which] = texUV;
}

void Mesh::setMaterial(Material *mat) { this->mat = mat; }

void Mesh::addFace(int v0, int v1, int v2) {
  GVT_ASSERT(v0 - 1 < vertices.size(), "Vertex index 0 outside bounds : " << (v0 - 1));
  GVT_ASSERT(v1 - 1 < vertices.size(), "Vertex index 1 outside bounds : " << (v1 - 1));
  GVT_ASSERT(v2 - 1 < vertices.size(), "Vertex index 2 outside bounds : " << (v2 - 1));
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
  for (int i = 0; i < normals.size(); ++i) normals[i] = Vector4f(0.0f, 0.0f, 0.0f, 0.0f);

  for (int i = 0; i < faces.size(); ++i) {
    int I = faces[i].get<0>();
    int J = faces[i].get<1>();
    int K = faces[i].get<2>();
    Vector4f a = vertices[I];
    Vector4f b = vertices[J];
    Vector4f c = vertices[K];
    Vector4f u = b - a;
    Vector4f v = c - a;
    Vector4f normal;
    normal.n[0] = u.n[1] * v.n[2] - u.n[2] * v.n[1];
    normal.n[1] = u.n[2] * v.n[0] - u.n[0] * v.n[2];
    normal.n[2] = u.n[0] * v.n[1] - u.n[1] * v.n[0];
    normal.n[3] = 0.0f;
    normal.normalize();
    face_normals[i] = normal;
    normals[I] += normal;
    normals[J] += normal;
    normals[K] += normal;
    faces_to_normals[i] = FaceToNormals(I, J, K);
  }
  for (int i = 0; i < normals.size(); ++i) normals[i].normalize();
  haveNormals = true;
}

Color Mesh::shadeFace(const int face_id, const Ray &r, const Vector4f &normal, const Light *lsource) {
  // XXX TODO: shadeFace returns constant color, fix?

  if (!faces_to_materials.size()) return shade(r, normal, lsource);

  Color c(0.5f, 0.5f, 0.5f, 0.0f);
  Material *m = (faces_to_materials[face_id] ? faces_to_materials[face_id] : mat);
  if (m) {
    c = m->shade(r, normal, lsource);
  }
  return c;
}

Color Mesh::shade(const Ray &r, const Vector4f &normal, const Light *lsource) { return mat->shade(r, normal, lsource); }

Box3D Mesh::computeBoundingBox() {

  Box3D box;
  for (int i = 0; i < vertices.size(); i++) {

    Point4f p = vertices[i];

    box.expand(p);
  }

  this->boundingBox = box;

  return box;
}
