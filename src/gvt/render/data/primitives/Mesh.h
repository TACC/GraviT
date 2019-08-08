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

#include <gvt/render/data/primitives/Data.h>
#include <gvt/render/data/primitives/Material.h>
#include <gvt/render/data/scene/Light.h>
#include <vector>


namespace gvt {
namespace render {
namespace data {
namespace primitives {

/// geometric mesh
/** geometric mesh used within geometric domains
\sa GeometryDomain
*/
//class Mesh : public AbstractMesh {
class Mesh : public Data {
public:
  typedef std::tuple<int, int, int> Face;
  typedef std::tuple<int, int, int, int> TetrahedralCell;
  typedef std::tuple<int, int, int> FaceToNormals;

  Mesh(gvt::render::data::primitives::Material *mat = NULL);
  Mesh(const Mesh &orig);

  virtual ~Mesh();
  virtual void setVertex(int which, glm::vec3 vertex, glm::vec3 normal = glm::vec3(), glm::vec3 texUV = glm::vec3());
  virtual void setNormal(int which, glm::vec3 normal = glm::vec3());
  virtual void setTexUV(int which, glm::vec3 texUV = glm::vec3());
  virtual void setMaterial(Material *mat);

  virtual void addVertexNormalTexUV(glm::vec3 vertex, glm::vec3 normal = glm::vec3(), glm::vec3 texUV = glm::vec3());
  virtual void addVertex(glm::vec3 vertex);
  virtual void addVertexColor(glm::vec3 color) { vertex_colors.push_back(color); }
  virtual void addNormal(glm::vec3 normal);
  virtual void addTexUV(glm::vec3 texUV);
  virtual void addFace(int v0, int v1, int v2);
  virtual void addTetrahedralCell(int v0, int v1, int v2, int v3);
  virtual void addFaceToNormals(FaceToNormals);

  virtual gvt::render::data::primitives::Box3D computeBoundingBox();
  virtual gvt::render::data::primitives::Box3D *getBoundingBox() { return &boundingBox; }
  virtual void generateNormals();

  virtual gvt::render::data::primitives::Material *getMaterial() { return mat; }

  void writeobj(std::string filename);

  virtual std::shared_ptr<Data> getData() {
    return std::shared_ptr<Data>(this);
  }


public:
  gvt::render::data::primitives::Material *mat;
  gvt::core::Vector<glm::vec3> vertices;
  gvt::core::Vector<glm::vec3> vertex_colors;
  gvt::core::Vector<glm::vec3> mapuv;
  gvt::core::Vector<glm::vec3> normals;
  gvt::core::Vector<Face> faces;
  gvt::core::Vector<TetrahedralCell> tets;
  gvt::core::Vector<FaceToNormals> faces_to_normals;
  gvt::core::Vector<glm::vec3> face_normals;
  gvt::core::Vector<Material *> faces_to_materials;
  std::vector<Material *> materials;
  gvt::render::data::primitives::Box3D boundingBox;
  bool haveNormals;
};
}
}
}
}

#endif /* GVT_RENDER_DATA_PRIMITIVES_MESH_H */
