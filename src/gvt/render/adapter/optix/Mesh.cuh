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
 * File:   Mesh.cuh
 * Author: Roberto Ribeiro
 *
 * Created on February 4, 2016, 11:00 PM
 */

#ifndef GVT_RENDER_DATA_PRIMITIVES_MESH_CUH
#define GVT_RENDER_DATA_PRIMITIVES_MESH_CUH

#include <vector_functions.h>
#include "Material.cuh"

namespace gvt {
namespace render {
namespace data {
namespace cuda_primitives {


class Matrix3f {
  //---[ Private Variable Declarations ]-----------------
public:
  // matrix elements in row major order

	float n[3][3];

  inline __device__ float3 operator*(const float3& v)
  {

  return make_float3(n[0][0] * v.x + n[1][0] * v.y + n[2][0] * v.z,
	  			n[0][1] * v.x + n[1][1] * v.y + n[2][1] * v.z ,
	  			n[0][2] * v.x + n[1][2] * v.y + n[2][2] * v.z );
  }

};


class Matrix4f {
  //---[ Private Variable Declarations ]-----------------
public:
  // matrix elements in row major order

	float n[4][4];


  inline __device__ float4 operator*(const float4& v)
  {

	  return make_float4(n[0][0] * v.x + n[1][0] * v.y + n[2][0] * v.z + n[3][0] * v.w,
	  			n[0][1] * v.x + n[1][1] * v.y + n[2][1] * v.z + n[3][1] * v.w,
	  			n[0][2] * v.x + n[1][2] * v.y + n[2][2] * v.z + n[3][2] * v.w,
	  			n[0][3] * v.x + n[1][3] * v.y + n[2][3] * v.z + n[3][3] * v.w);

  }


};

/// base class for mesh
class AbstractMesh {
public:
	 __device__ AbstractMesh() {}

	 __device__ AbstractMesh(const AbstractMesh &) {}

	 __device__  ~AbstractMesh() {}

};

/// geometric mesh
/** geometric mesh used within geometric domains
\sa GeometryDomain
*/

class Mesh : public AbstractMesh {
public:

   __device__ Mesh(){

  }

   __device__ virtual ~Mesh(){

  }

public:
   data::primitives::Material* mat;
  cuda_vec* vertices;
  cuda_vec* normals;
  int3* faces;
  int3* faces_to_normals;
  cuda_vec* face_normals;

};
}
}
}
}

#endif /* GVT_RENDER_DATA_PRIMITIVES_MESH_CUH */
