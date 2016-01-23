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
 * File:   readply.cpp
 * Author: jbarbosa
 *
 * Created on April 22, 2014, 10:24 AM
 */

#include <gvt/render/data/domain/reader/PlyReader.h>

#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>

// Manta includes
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
// end Manta includes

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace gvt::render::data::domain::reader;
using namespace gvt::render::data::primitives;

PlyReader::PlyReader(std::string filename) {

  plyMesh = new Mesh();
  Manta::Mesh *mesh = new Manta::Mesh();
  Manta::Material *material = new Manta::Phong(Manta::Color::white() * 0.6, Manta::Color::white() * 0.8, 16, 0);
  Manta::MeshTriangle::TriangleType triangleType = Manta::MeshTriangle::KENSLER_SHIRLEY_TRI;
  readPlyFile(filename, Manta::AffineTransform::createIdentity(), mesh, material, triangleType);

  for (int i = 0; i < mesh->vertices.size(); i++) {

    Manta::Vector v = mesh->vertices[i];
    gvt::core::math::Vector4f vertex(v[0], v[1], v[2], 1.f);
    plyMesh->addVertex(vertex);
  }

  for (int i = 0; i < mesh->vertex_indices.size(); i += 3) {
    plyMesh->addFace(mesh->vertex_indices[i], mesh->vertex_indices[i + 1], mesh->vertex_indices[i + 2]);
  }

  delete material;
}

PlyReader::~PlyReader() {}