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
 * File:   Mesh.h
 * Author: jbarbosa
 *
 * Created on April 20, 2014, 11:01 PM
 */

#ifndef GVT_RENDER_DATA_PRIMITIVES_MESH_H
#define GVT_RENDER_DATA_PRIMITIVES_MESH_H

#include <gvt/render/data/primitives/BBox.h>
#include <gvt/render/data/primitives/Material.h>
#include <gvt/render/data/scene/Light.h>

#include <vector>

#include <boost/container/vector.hpp>
#include <boost/tuple/tuple.hpp>

namespace gvt {
namespace render {
namespace data {
namespace primitives {

/// base class for mesh
class AbstractMesh {
public:
  AbstractMesh() {}

  AbstractMesh(const AbstractMesh &) {}

  ~AbstractMesh() {}

  virtual AbstractMesh *getMesh() { return this; }

  virtual gvt::render::data::primitives::Box3D *getBoundingBox() { return NULL; }
};

/// geometric mesh
/** geometric mesh used within geometric domains
\sa GeometryDomain
*/
class Mesh : public AbstractMesh {
public:
  typedef boost::tuple<int, int, int> Face;
  typedef boost::tuple<int, int, int> FaceToNormals;

  Mesh(gvt::render::data::primitives::Material *mat = NULL);
  Mesh(const Mesh &orig);

  virtual ~Mesh();
  virtual void setVertex(int which, glm::vec3 vertex, glm::vec3 normal = glm::vec3(), glm::vec3 texUV = glm::vec3());
  virtual void setNormal(int which, glm::vec3 normal = glm::vec3());
  virtual void setTexUV(int which, glm::vec3 texUV = glm::vec3());
  virtual void setMaterial(Material *mat);

  virtual void addVertexNormalTexUV(glm::vec3 vertex, glm::vec3 normal = glm::vec3(), glm::vec3 texUV = glm::vec3());
  virtual void addVertex(glm::vec3 vertex);
  virtual void addNormal(glm::vec3 normal);
  virtual void addTexUV(glm::vec3 texUV);
  virtual void addFace(int v0, int v1, int v2);
  virtual void addFaceToNormals(FaceToNormals);

  virtual gvt::render::data::primitives::Box3D computeBoundingBox();
  virtual gvt::render::data::primitives::Box3D *getBoundingBox() { return &boundingBox; }
  virtual void generateNormals();

  virtual gvt::render::data::primitives::Material *getMaterial() { return mat; }
//  virtual gvt::render::data::Color shade(const gvt::render::actor::Ray &r, const glm::vec3 &normal,
//                                         const gvt::render::data::scene::Light *lsource,
//                                         const glm::vec3 areaLightPosition);
//
//  virtual gvt::render::data::Color shadeFace(const int face_id, const gvt::render::actor::Ray &r,
//                                             const glm::vec3 &normal, const gvt::render::data::scene::Light *lsource);
//
//  virtual gvt::render::data::Color shadeFaceAreaLight(const int face_id, const gvt::render::actor::Ray &r,
//                                                      const glm::vec3 &normal,
//                                                      const gvt::render::data::scene::Light *lsource,
//                                                      const glm::vec3 areaLightPosition);

  void writeobj(std::string filename);

public:
  gvt::render::data::primitives::Material *mat;
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec3> mapuv;
  std::vector<glm::vec3> normals;
  std::vector<Face> faces;
  std::vector<FaceToNormals> faces_to_normals;
  std::vector<glm::vec3> face_normals;
  std::vector<Material *> faces_to_materials;
  gvt::render::data::primitives::Box3D boundingBox;
  bool haveNormals;
};
}
}
}
}

#endif /* GVT_RENDER_DATA_PRIMITIVES_MESH_H */
