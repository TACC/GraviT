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
 * File:   TinyObjReaderAdapter.cpp
 * Author: vchen
 *
 * Created on April 22, 2014, 10:24 AM
 */

#include <gvt/render/data/domain/reader/TinyObjReaderAdapter.h>
#include <tiny_obj_loader.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace gvt::core::math;
using namespace gvt::render::data::domain::reader;
using namespace gvt::render::data::primitives;



TinyObjReaderAdapter::TinyObjReaderAdapter(std::string filename, std::string materialsBasePath) {
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string err;

  tinyobj::LoadObj(shapes, materials, filename.c_str(),materialsBasePath.c_str());

  this->objMeshes = new Mesh * [shapes.size()];
  this->meshbboxes = new Box3D * [shapes.size()];
  this->meshCount = shapes.size();
  
  for (size_t i = 0; i < shapes.size(); i++) {
    // get the material index
    int materialIndex = shapes[i].mesh.material_ids[0];
  
      // check the illumination model
    int illuminationModel = materials[materialIndex].illum;
    gvt::render::data::primitives::Mesh * newMesh;

    switch (illuminationModel) {
      case 5:
        //temp this should be a mirror finish or something else.
        newMesh =  new Mesh(new Phong(
                            Vector4f(materials[materialIndex].diffuse[0], materials[materialIndex].diffuse[1], materials[materialIndex].diffuse[2], 1.0),
                            Vector4f(materials[materialIndex].specular[0], materials[materialIndex].specular[1], materials[materialIndex].specular[2], 1.0),1.0));
        break;

      default:
        newMesh =  new Mesh(new Phong(
                              Vector4f(materials[materialIndex].diffuse[0], materials[materialIndex].diffuse[1], materials[materialIndex].diffuse[2], 1.0),
                              Vector4f(materials[materialIndex].specular[0], materials[materialIndex].specular[1], materials[materialIndex].specular[2], 1.0),1.0));
    }

    GVT_ASSERT((shapes[i].mesh.positions.size() % 3) == 0,"Mesh verticies is incorrect.");
    
    Point4f lower,upper;
    //create verticies
    for (size_t v = 0; v < shapes[i].mesh.positions.size() / 3; v++) {
      Point4f newPoint (shapes[i].mesh.positions[3*v+0],
                        shapes[i].mesh.positions[3*v+1],
                        shapes[i].mesh.positions[3*v+2],
                        1.0);
      if (v == 0)
      {
        lower = newPoint;
        upper = newPoint;
      }
      else
      {
        for (int j = 0; j < 3; j++) {
          lower[j] = (lower[j] < newPoint[j]) ? lower[j] : newPoint[j];
          upper[j] = (upper[j] > newPoint[j]) ? upper[j] : newPoint[j];
        }
      }
      newMesh->addVertex(newPoint);
    }
    //bb box
    Box3D *meshbbox = new gvt::render::data::primitives::Box3D(lower, upper);
    this->meshbboxes[i] = meshbbox;

    //create faces, tinyobj converts polygons into triangles
    for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
      //shapes[i].mesh.material_ids[f]
      //1 based index
      newMesh->addFace(shapes[i].mesh.indices[3*f+0] +1, shapes[i].mesh.indices[3*f+1] +1, shapes[i].mesh.indices[3*f+2]+1);
       }
    newMesh->generateNormals();
    newMesh->computeBoundingBox();
    this->objMeshes[i] = newMesh;
  }

};

TinyObjReaderAdapter::~TinyObjReaderAdapter() {}
