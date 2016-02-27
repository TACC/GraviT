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
 * File:   ObjReader.h
 * Author: jbarbosa
 *
 * Created on January 22, 2015, 1:36 PM
 */

#ifndef GVT_RENDER_DATA_DOMAIN_READER_OBJ_READER_H
#define GVT_RENDER_DATA_DOMAIN_READER_OBJ_READER_H

#include <gvt/render/data/Primitives.h>

#include <string>
#include <vector>

namespace gvt {
namespace render {
namespace data {
namespace domain {
namespace reader {
/// GVT ObjReader - file reader for files in Wavefront .obj format
/** This class contains methods for parsing and reading files in .obj format.
 * The data in the file consists of vertex and face specifications
 * as well as normals, textures, and material specifications. The data is stored
 * in a gvt Mesh object for processing by the gvt software.
*
*/
class ObjReader {
public:
  /** Constructor opens the given file and parses it line by line placing data in the mesh
  *   object.
  */
  ObjReader(const std::string filename = "");
  virtual ~ObjReader();

  /** public member function to return a pointer to the mesh object.
  */
  gvt::render::data::primitives::Mesh *getMesh() { return objMesh; }

private:
  /** private member function to parse a line from the .obj file corresponding to a geometric vertex.
  *   vertices are signified in the file by a line with the first non whitespece character being
  *   a 'v'. This is followed by x, y, z, and an optional weight w. Weights are only used for rational
  *   curves and surfaces which are not supported by the current system. A weight in the geometric
  *   vertex data will cause an error message to be written by the parser.
  */
  void parseVertex(std::string line);
  /** private member function to parse a vertex normal. Vertex normals are specified by placing a
  *   'vn' as the first non whitespace characters on a line. This is followed by the i, j, and k
  *   components of the vector. The three components are floating point numbers.
  */
  void parseVertexNormal(std::string line);
  /** private member function to parse texture vertex data. Texture coordinates are specified as three
  *   floating point numbers which follow the 'vt' designator as the first non whitespace characters
  *   on a line. The three floats u, v, and w, represent texture coordinates for the particular vertex
  */
  void parseVertexTexture(std::string line);
  /** private member function to parse polygonal face data. Face data is present on a line beginning
  *   with a 'f' in the first non white space character position. This is followed by collections
  *   of indices into the vertex list, and optionally into the texture and normal list as well. A
  *   face can consist of 3 or more vertices and possibly corresponding texture and normal data. If
  *   present the texture index follows the normal index separated by a '/'. If there is also normal
  *   data it follows the texture data separated from it by a '/'. An entry for a vertex may consist
  *   of any of the following forms where # represents the number of the vertex for this particular
  *   face, v# for a single vertex, v#/vt# for vertex and texture index, v#/vt#/vn# for vertex with
  *   texture and normal data, and finally v#//vn# for vertex with normal data and no texture data.
  */
  void parseFace(std::string line);
  /** Pointer to the member data mesh object that the data is stored in. Data values filled in by
  *   the constructor.
  */
  gvt::render::data::primitives::Mesh *objMesh;
  /** Logical member variable indicating whether to compute normals from vertex data. Initial
  *   value false.
  */
  bool computeNormals;
};
}
}
}
}
}

#endif /* GVT_RENDER_DATA_DOMAIN_READER_OBJ_READER_H */
