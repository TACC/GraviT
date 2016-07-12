
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
 * File:   Material.h
 * Author: jbarbosa
 *
 * Created on April 18, 2014, 3:07 PM
 */

#ifndef GVT_RENDER_DATA_PRIMITIVES_MATERIAL_H
#define GVT_RENDER_DATA_PRIMITIVES_MATERIAL_H

//#include <stdio.h>
#include <glm/glm.hpp>

//#include <cmath>
//#include <gvt/render/data/DerivedTypes.h>


namespace gvt {
namespace render {
namespace data {
namespace primitives {

typedef enum {
    LAMBERT,
    PHONG,
    BLINN,
    EMBREE_MATERIAL_METAL,
    EMBREE_MATERIAL_VELVET,
    EMBREE_MATERIAL_MATTE
  } MATERIAL_TYPE;


struct Material {

  //Default material
  Material(){
    type = LAMBERT;
    //type = EMBREE_MATERIAL_MATTE;
    kd = glm::vec3(.5,.5,.5);
    ks = glm::vec3(.5,.5,.5);
    alpha = 1.f;

    //type = EMBREE_MATERIAL_METAL;
    //copper metal
    eta = glm::vec3(.19,1.45, 1.50);
    k = glm::vec3(3.06,2.40, 1.88);
    roughness = 0.05;
  }

  int type;

  glm::vec3 ks; //diffuse k
  glm::vec3 kd; // specular k
  float alpha;
  glm::vec3 eta;//EmbreeMetalMaterial
  glm::vec3 k; //EmbreeMetalMaterial
  float roughness; //EmbreeMetalMaterial
  glm::vec3 horizonScatteringColor; //EmbreeVelvetMaterial
  float backScattering; //EmbreeVelvetMaterial
  float horizonScatteringFallOff; //EmbreeVelvetMaterial

};


}
}
}
}

#endif /* GVT_RENDER_DATA_PRIMITIVES_MATERIAL_H */

