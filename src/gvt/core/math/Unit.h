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
#ifndef Math_Unit_h
#define Math_Unit_h
/*
    This file is part of Magnum.

    Copyright © 2010, 2011, 2012, 2013, 2014
              Vladimír Vondruš <mosra@centrum.cz>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*/

/** @file
 * @brief Class Magnum::Math::Constants
 */

namespace gvt {
namespace core {
namespace math {

/**
@brief Base class for units
@tparam T Underlying data type

@see Deg, Rad
*/
template <template <class> class Derived, class T> class Unit {
  template <template <class> class, class> friend class Unit;

public:
  typedef T Type; /**< @brief Underlying data type */

  Unit() : value(T(0)) {}

  explicit Unit(T value) : value(value) {}

  template <class U> explicit Unit(Unit<Derived, U> value) : value(T(value.value)) {}

  operator T() const { return value; }

  bool operator==(Unit<Derived, T> other) const { return value == other.value; }

  bool operator!=(Unit<Derived, T> other) const { return !operator==(other); }

  bool operator<(Unit<Derived, T> other) const { return value < other.value; }

  bool operator>(Unit<Derived, T> other) const { return value > other.value; }

  bool operator<=(Unit<Derived, T> other) const { return !operator>(other); }

  bool operator>=(Unit<Derived, T> other) const { return !operator<(other); }

  Unit<Derived, T> operator-() const { return Unit<Derived, T>(-value); }

  Unit<Derived, T> &operator+=(Unit<Derived, T> other) {
    value += other.value;
    return *this;
  }
  Unit<Derived, T> operator+(Unit<Derived, T> other) const { return Unit<Derived, T>(value + other.value); }

  Unit<Derived, T> &operator-=(Unit<Derived, T> other) {
    value -= other.value;
    return *this;
  }

  Unit<Derived, T> operator-(Unit<Derived, T> other) const { return Unit<Derived, T>(value - other.value); }

  Unit<Derived, T> &operator*=(T number) {
    value *= number;
    return *this;
  }

  Unit<Derived, T> operator*(T number) const { return Unit<Derived, T>(value * number); }

  Unit<Derived, T> &operator/=(T number) {
    value /= number;
    return *this;
  }

  Unit<Derived, T> operator/(T number) const { return Unit<Derived, T>(value / number); }

  T operator/(Unit<Derived, T> other) const { return value / other.value; }

private:
  T value;
};
}
}
}

#endif
