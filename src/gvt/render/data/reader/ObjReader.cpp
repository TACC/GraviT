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
 * File:   ObjReader.cpp
 * Author: jbarbosa
 *
 * Created on January 22, 2015, 1:36 PM
 */

#include <gvt/render/data/reader/ObjReader.h>

#include <gvt/core/Debug.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include <tiny_obj_loader.h>

using namespace gvt::render::data::domain::reader;
using namespace gvt::render::data::primitives;

namespace gvt {
namespace render {
namespace data {
namespace domain {
namespace reader {
std::vector<std::string> split(const std::string &s, char delim, std::vector<std::string> &elems) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (item.length() > 0) {
      elems.push_back(item);
    }
  }
  return elems;
}
}
}
}
}
} // namespace reader} namespace domain} namespace data} namespace render} namespace gvt}

ObjReader::ObjReader(const std::string filename) : computeNormals(false) {

  // GVT_ASSERT(filename.size() > 0, "Invalid filename");
  // std::fstream file;
  // file.open(filename.c_str());
  // GVT_ASSERT(file.good(), "Error loading obj file " << filename);

  Material* m = new Material();
//  m->type = LAMBERT;
//  //m->type = EMBREE_MATERIAL_MATTE;
//  m->kd = glm::vec3(1.0,1.0, 1.0);
//  m->ks = glm::vec3(1.0,1.0,1.0);
//  m->alpha = 0.5;
//
//  //m->type = EMBREE_MATERIAL_METAL;
//  //copper metal
//  m->eta = glm::vec3(.19,1.45, 1.50);
//  m->k = glm::vec3(3.06,2.40, 1.88);
//  m->roughness = 0.05;

  objMesh = new Mesh(m);

  // while (file.good()) {
  //   std::string line;
  //   std::getline(file, line);
  //
  //   if (line.find("#") == 0) continue;
  //
  //   if (line.find("v") == 0) {
  //     parseVertex(line);
  //     continue;
  //   }
  //   if (line.find("vn") == 0) {
  //     parseVertexNormal(line);
  //     continue;
  //   }
  //   if (line.find("vt") == 0) {
  //     parseVertexTexture(line);
  //     continue;
  //   }
  //   if (line.find("f") == 0) {
  //     parseFace(line);
  //     continue;
  //   }
  // }

  std::size_t found = filename.find_last_of("/");
  std::string path = filename.substr(0, found + 1);

  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  // std::vector<pathtr::math::vec3<float> > vertices;
  // std::vector<pathtr::math::vec3<int> > faces;
  // std::vector<pathtr::math::vec3<float> > normals;
  // std::vector<size_t> cindex;
  // std::vector<pathtr::material> color;
  // std::vector<size_t> lfaces;
  // bbox bb;

  std::string err;

  if (!tinyobj::LoadObj(shapes, materials, err, filename.c_str(), path.c_str())) {
    std::cerr << err << std::endl;
    exit(1);
  }

  // color.resize(materials.size());
  // ;
  // for (size_t i = 0; i < materials.size(); i++) {
  //   pathtr::material &m = color[i];
  //   if (materials[i].name.find("light") != std::string::npos) {
  //     m._light = true;
  //   }
  //
  //   m.illum = materials[i].illum;
  //
  //   m.ambient = pathtr::rgb(materials[i].ambient[0], materials[i].ambient[1], materials[i].ambient[2]);
  //   m.diffuse = pathtr::rgb(materials[i].diffuse[0], materials[i].diffuse[1], materials[i].diffuse[2]);
  //   m.specular = pathtr::rgb(materials[i].specular[0], materials[i].specular[1], materials[i].specular[2]);
  //   m.transmittance =
  //       pathtr::rgb(materials[i].transmittance[0], materials[i].transmittance[1], materials[i].transmittance[2]);
  //   m.emission = pathtr::rgb(materials[i].emission[0], materials[i].emission[1], materials[i].emission[2]);
  //   m.shininess = materials[i].shininess;
  //   m.ior = materials[i].ior;
  //   m.dissolve = materials[i].dissolve;
  // }

  size_t vertices_offset = 0;

  for (size_t i = 0; i < shapes.size(); i++) {

    for (size_t v = 0; v < shapes[i].mesh.positions.size() / 3; v++) {

      objMesh->vertices.push_back(glm::vec3(shapes[i].mesh.positions[3 * v + 0], shapes[i].mesh.positions[3 * v + 1],
                                            shapes[i].mesh.positions[3 * v + 2]));

      objMesh->boundingBox.expand(objMesh->vertices[objMesh->vertices.size() - 1]);

      if (!shapes[i].mesh.normals.empty()) {
        glm::vec3 n(shapes[i].mesh.normals[3 * v + 0], shapes[i].mesh.normals[3 * v + 1],
                    shapes[i].mesh.normals[3 * v + 2]);

        n = glm::normalize(n);

        objMesh->normals.push_back(n);
      }
    }

    for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
      objMesh->faces.push_back(Mesh::Face(vertices_offset + shapes[i].mesh.indices[3 * f + 0],
                                          vertices_offset + shapes[i].mesh.indices[3 * f + 1],
                                          vertices_offset + shapes[i].mesh.indices[3 * f + 2]));

      // size_t midx = shapes[i].mesh.material_ids[f];
      // cindex.push_back(midx);
      // if (color[midx].light()) {
      //   lfaces.push_back(faces.size() - 1);
      // }
    }

    vertices_offset += shapes[i].mesh.positions.size() / 3;
  }

  computeNormals = (objMesh->normals.size() == objMesh->vertices.size());
  std::cout << "Found : " << objMesh->vertices.size() << " vertices" << std::endl;
  std::cout << "Found : " << objMesh->normals.size() << " normals" << std::endl;
  std::cout << "Found : " << objMesh->faces.size() << " normals" << std::endl;
  std::cout << "Bound : " << objMesh->boundingBox.bounds_min << " x " << objMesh->boundingBox.bounds_max << std::endl;
  std::cout << "Center : " << ((objMesh->boundingBox.bounds_min + objMesh->boundingBox.bounds_max) * .5f) << std::endl;

  if (computeNormals) objMesh->generateNormals();
  objMesh->computeBoundingBox();
}

void ObjReader::parseVertex(std::string line) {
  std::vector<std::string> elems;
  split(line, ' ', elems);
  GVT_ASSERT(elems.size() == 4, "Error parsing vertex");
  objMesh->addVertex(glm::vec3(std::atof(elems[1].c_str()), std::atof(elems[2].c_str()), std::atof(elems[3].c_str())));
}

void ObjReader::parseVertexNormal(std::string line) {
  std::vector<std::string> elems;
  split(line, ' ', elems);
  GVT_ASSERT(elems.size() == 4, "Error parsing vertex normal");
  objMesh->addNormal(glm::vec3(std::atof(elems[1].c_str()), std::atof(elems[2].c_str()), std::atof(elems[3].c_str())));
}

void ObjReader::parseVertexTexture(std::string line) {
  std::vector<std::string> elems;
  split(line, ' ', elems);
  GVT_ASSERT(elems.size() == 3, "Error parsing texture map");
  objMesh->addTexUV(glm::vec3(std::atof(elems[1].c_str()), std::atof(elems[2].c_str()), 0));
}

void ObjReader::parseFace(std::string line) {
  std::vector<std::string> elems;
  split(line, ' ', elems);
  GVT_ASSERT(elems.size() == 4, "Error parsing face");

  int v1, n1, t1;
  int v2, n2, t2;
  int v3, n3, t3;

  v1 = n1 = t1 = 0;
  v2 = n2 = t2 = 0;
  v3 = n3 = t3 = 0;

  /* can be one of %d, %d//%d, %d/%d, %d/%d/%d %d//%d */
  if (std::strstr(elems[1].c_str(), "//")) {
    std::sscanf(elems[1].c_str(), "%d//%d", &v1, &n1);
    std::sscanf(elems[2].c_str(), "%d//%d", &v2, &n2);
    std::sscanf(elems[3].c_str(), "%d//%d", &v3, &n3);
    objMesh->addFaceToNormals(Mesh::FaceToNormals(n1 - 1, n2 - 1, n3 - 1));
  } else if (std::sscanf(elems[1].c_str(), "%d/%d/%d", &v1, &t1, &n1) == 3) {
    /* v/t/n */
    std::sscanf(elems[2].c_str(), "%d/%d/%d", &v2, &t2, &n2);
    std::sscanf(elems[3].c_str(), "%d/%d/%d", &v3, &t3, &n3);
    objMesh->addFace(v1, v2, v3);
    objMesh->addFaceToNormals(Mesh::FaceToNormals(n1 - 1, n2 - 1, n3 - 1));
  } else if (std::sscanf(elems[1].c_str(), "%d/%d", &v1, &t1) == 2) {
    /* v/t */
    std::sscanf(elems[2].c_str(), "%d/%d", &v2, &t2);
    std::sscanf(elems[3].c_str(), "%d/%d", &v3, &t3);
    objMesh->addFace(v1, v2, v3);
    computeNormals = true;
  } else {
    /* v */
    std::sscanf(elems[1].c_str(), "%d", &v1);
    std::sscanf(elems[2].c_str(), "%d", &v2);
    std::sscanf(elems[3].c_str(), "%d", &v3);
    computeNormals = true;
  }

  objMesh->addFace(v1, v2, v3);
}

ObjReader::~ObjReader() {}
