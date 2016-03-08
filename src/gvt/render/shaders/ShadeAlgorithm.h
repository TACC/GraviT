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
 * Shader.h
 *
 *  Created on: Mar 6, 2016
 *      Author: Roberto Ribeiro
 */

#ifndef GVT_RENDER_SHADER_H
#define GVT_RENDER_SHADER_H

#include <gvt/render/data/DerivedTypes.h>
#include <gvt/render/data/primitives/Mesh.h>
#include <gvt/render/actor/Ray.h>

using namespace gvt::render::actor;
using namespace gvt::render::data::primitives;


namespace gvt {
namespace render {
namespace shader {

class ShadeAlgorithm {
public:

  ShadeAlgorithm() {};

  virtual ~ShadeAlgorithm() {}

  virtual bool shade(Ray& r, const glm::vec3 &normal,gvt::render::data::primitives::UnifiedMateral * material,
 		   TLRand& randEngine, gvt::render::actor::RayVector& shadowRays,  int* valid)=0;

};

}
} // render
} // gvt

#endif // GVT_RENDER_SHADER_H
