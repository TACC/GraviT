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

//#include <gvt/render/data/primitives/BBox.h>
//#include <gvt/render/data/primitives/Material.h>
//#include <gvt/render/data/scene/Light.h>

//#include <gvt/render/data/primitives/Mesh.h>

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
  float n[9];


  inline __device__ float3 operator*(const float3& v)
  {
      return make_float3(n[0] * v.x + n[1] * v.y + n[2] * v.z,
              n[3] * v.x + n[4] * v.y + n[5] * v.z,
              n[6] * v.x + n[7] * v.y + n[8] * v.z);
  }

};


class Matrix4f {
  //---[ Private Variable Declarations ]-----------------
public:
  // matrix elements in row major order
  float n[16];

/*
  inline __device__ float3 operator*(const float3& v)
  {
      return make_float3(n[0] * v.x + n[1] * v.y + n[2] * v.z,
              n[3] * v.x + n[4] * v.y + n[5] * v.z,
              n[6] * v.x + n[7] * v.y + n[8] * v.z);
  }*/

  inline __device__ float4 operator*(const float4& v)
  {
	  return make_float4(n[0] * v.x + n[1] * v.y + n[2] * v.z + n[3] * v.w,
              n[4] * v.x + n[5] * v.y + n[6] * v.z + n[7] * v.w,
              n[8] * v.x + n[9] * v.y + n[10] * v.z + n[11] * v.w,
              n[12] * v.x + n[13] * v.y + n[14] * v.z + n[15] * v.w);
  }


};

/// base class for mesh
class AbstractMesh {
public:
	 __device__ AbstractMesh() {}

	 __device__ AbstractMesh(const AbstractMesh &) {}

	 __device__  ~AbstractMesh() {}

 // virtual gvt::render::data::primitives::Box3D *getBoundingBox() { return NULL; }
};

/// geometric mesh
/** geometric mesh used within geometric domains
\sa GeometryDomain
*/

class Mesh : public AbstractMesh {
public:
  //typedef boost::tuple<int, int, int> Face;
  //typedef boost::tuple<int, int, int> FaceToNormals;
   __device__ Mesh(){

  }

   __device__ virtual ~Mesh(){

  }
  //  virtual gvt::render::data::primitives::Box3D *getBoundingBox() { return &boundingBox; }

  /*virtual gvt::render::data::primitives::Material *getMaterial() { return mat; }
  virtual gvt::render::data::Color shade(const gvt::render::actor::Ray &r, const gvt::core::math::Vector4f &normal,
                                         const gvt::render::data::scene::Light *lsource);

  virtual gvt::render::data::Color shadeFace(const int face_id, const gvt::render::actor::Ray &r,
                                             const gvt::core::math::Vector4f &normal,
                                             const gvt::render::data::scene::Light *lsource);
*/
public:
   Material* mat;
  //float4* vertices;
  //float4* mapuv;
  float4* normals;
  int3* faces;
  int3* faces_to_normals;
  float4* face_normals;
  //float4* faces_to_materials;
  //gvt::render::data::primitives::Box3D boundingBox;
  //bool haveNormals;
};
}
}
}
}

#endif /* GVT_RENDER_DATA_PRIMITIVES_MESH_CUH */
