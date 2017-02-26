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
 * File:   Math.h
 * Author: jbarbosa
 *
 * Created on March 28, 2014, 1:15 PM
 */

#ifndef GVT_CORE_MATH_H
#define GVT_CORE_MATH_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat3x4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <iostream>

#include <gvt/core/math/RandEngine.h>

#define GVT_ALIGN(x) __attribute__((aligned(x)))

namespace glm {
inline std::ostream &operator<<(std::ostream &os, const glm::vec3 &v) {
  return os << "{" << v[0] << ", " << v[1] << ", " << v[2] << "}";
}
inline std::ostream &operator<<(std::ostream &os, const glm::vec4 &v) {
  return os << "{" << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << "}";
}
inline std::ostream &operator<<(std::ostream &os, const glm::mat4 &v) {
  return os << "{" << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << "}";
}
inline std::ostream &operator<<(std::ostream &os, const glm::mat4 *v) {
  return os << "{" << (*v)[0] << ", " << (*v)[1] << ", " << (*v)[2] << ", " << (*v)[3] << "}";
}
inline std::ostream &operator<<(std::ostream &os, const glm::mat3 &v) {
  return os << "{" << v[0] << ", " << v[1] << ", " << v[2] << "}";
}
inline std::ostream &operator<<(std::ostream &os, const glm::mat3 *v) {
  return os << "{" << (*v)[0] << ", " << (*v)[1] << ", " << (*v)[2] << "}";
}
};

#endif /* GVT_CORE_MATH_H */
