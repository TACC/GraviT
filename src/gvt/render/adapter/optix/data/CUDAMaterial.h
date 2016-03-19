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
 * File:   CUDAMaterial.h
 * Author: Roberto Ribeiro
 *
 * Created on February 4, 2016, 19:00 PM
 */

#ifndef GVT_RENDER_DATA_PRIMITIVES_CUDAMATERIAL_CUH
#define GVT_RENDER_DATA_PRIMITIVES_CUDAMATERIAL_CUH

#include <vector_functions.h>
#include <gvt/render/data/primitives/Material.h>

#include <glm/vec3.hpp>

namespace gvt {
namespace render {
namespace data {
namespace primitives {

struct CUDALambert {

  CUDALambert(const glm::vec3 &_kd) {
    kd = make_float4(_kd[0], _kd[1], _kd[2], 0.0f);
    type = CUDA_LAMBERT;
  }

  data::primitives::MATERIAL_TYPE type;
  float4 kd;
};

struct CUDAPhong {

  CUDAPhong(const glm::vec3 &_kd, const glm::vec3 &_ks, const float &_alpha) {
    kd = make_float4(_kd[0], _kd[1], _kd[2], 0.0f);
    ks = make_float4(_ks[0], _ks[1], _ks[2], 0.0f);
    alpha = _alpha;
    type = CUDA_PHONG;
  }

  data::primitives::MATERIAL_TYPE type;
  float4 kd;
  float4 ks;
  float alpha;
};

class CUDABlinnPhong {

  CUDABlinnPhong(const glm::vec3 &_kd, const glm::vec3 &_ks, const float &_alpha) {
    kd = make_float4(_kd[0], _kd[1], _kd[2], 0.0f);
    ks = make_float4(_ks[0], _ks[1], _ks[2], 0.0f);
    alpha = _alpha;
    type = CUDA_BLINN;
  }

  data::primitives::MATERIAL_TYPE type;
  float4 kd;
  float4 ks;
  float alpha;
};
}
}
}
}

#endif
