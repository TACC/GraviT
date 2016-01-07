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

#ifndef GVT_CORE_MATH_ANGLE_H
#define GVT_CORE_MATH_ANGLE_H

#include <gvt/core/math/Constants.h>
#include <gvt/core/math/Unit.h>

namespace gvt {
namespace core {
namespace math {
template <class T> class Deg;
template <class T> class Rad;

// angle in degrees
/**
templated class to represent an angle in degrees.
Templeted to allow both single and double precision representation.
*/
template <class T> class Deg : public Unit<Deg, T> {
public:
  Deg() {}

  explicit Deg(T value) : Unit<math::Deg, T>(value) {}

  Deg(Unit<math::Deg, T> value) : Unit<math::Deg, T>(value) {}

  template <class U> Deg(Unit<math::Deg, U> value) : Unit<math::Deg, T>(value) {}

  Deg(Unit<Rad, T> value);
};

// angle in radians
/**
templated class to represent an angle in radians.
Templeted to allow both single and double precision representation.
*/
template <class T> class Rad : public Unit<Rad, T> {
public:
  explicit Rad(T value) : Unit<math::Rad, T>(value) {}

  Rad(Unit<math::Rad, T> value) : Unit<math::Rad, T>(value) {}

  template <class U> explicit Rad(Unit<math::Rad, U> value) : Unit<math::Rad, T>(value) {}

  Rad(Unit<Deg, T> value);
};

template <class T> Deg<T>::Deg(Unit<Rad, T> value) : Unit<math::Deg, T>(T(180) * T(value) / Constants<T>::pi()) {}
template <class T> Rad<T>::Rad(Unit<Deg, T> value) : Unit<math::Rad, T>(T(value) * Constants<T>::pi() / T(180)) {}
}
}
}
#endif /* GVT_CORE_MATH_ANGLE_H */
