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
 * File:   Function.h
 * Author: jbarbosa
 *
 * Created on March 14, 2014, 5:54 PM
 */

#ifndef GVT_CORE_MATH_FUNCTION_H
#define GVT_CORE_MATH_FUNCTION_H

#include <gvt/core/math/Vector.h>

#include <cmath>
#include <limits>
#include <fstream>

namespace gvt {
namespace core {
namespace math {
template <class T> inline T sin(Unit<Rad, T> angle) { return std::sin(T(angle)); }
template <class T> inline T sin(Unit<Deg, T> angle) { return sin(Rad<T>(angle)); }
template <class T> inline T cos(Unit<Rad, T> angle) { return std::cos(T(angle)); }
template <class T> inline T cos(Unit<Deg, T> angle) { return cos(Rad<T>(angle)); }
template <class T> inline T tan(Unit<Rad, T> angle) { return std::tan(T(angle)); }
template <class T> inline T tan(Unit<Deg, T> angle) { return tan(Rad<T>(angle)); }
template <class T> inline Rad<T> asin(T value) { return Rad<T>(std::asin(value)); }
template <class T> inline Rad<T> acos(T value) { return Rad<T>(std::acos(value)); }
template <class T> inline Rad<T> atan(T value) { return Rad<T>(std::atan(value)); }
template <class T> inline T min(T a, T b) { return std::min(a, b); }
template <class T> inline T max(T a, T b) { return std::max(a, b); }
template <class T> inline std::pair<T, T> minmax(T a, T b) {
  return a < b ? std::make_pair(a, b) : std::make_pair(b, a);
}
template <class T> inline T abs(T a) { return std::abs(a); }
template <class T> inline T floor(T a) { return std::floor(a); }
template <class T> inline T ceil(T a) { return std::ceil(a); }
template <class T> inline T sqrt(T a) { return T(std::sqrt(a)); }
template <class T> inline T sqrtInverted(T a) { return T(1) / std::sqrt(a); }
template <std::size_t size, class T> Vector<size, T> sqrtInverted(const Vector<size, T> &a) {
  return Vector<size, T>(T(1)) / sqrt(a);
}
template <class T> inline T clamp(T value, T min, T max) { return std::min(std::max(value, min), max); }
template <class T, class U> inline T lerp(T a, T b, U t) { return T((U(1) - t) * a + t * b); }
template <class T> inline T lerpInverted(T a, T b, T lerp) { return (lerp - a) / (b - a); }

template <std::size_t size, class T> inline Vector<size, T> min(const Vector<size, T> &a, const Vector<size, T> &b) {
  Vector<size, T> out;
  for (std::size_t i = 0; i != size; ++i) out[i] = std::min(a[i], b[i]);
  return out;
}

template <std::size_t size, class T> Vector<size, T> max(const Vector<size, T> &a, const Vector<size, T> &b) {
  Vector<size, T> out;
  for (std::size_t i = 0; i != size; ++i) out[i] = std::max(a[i], b[i]);
  return out;
}

template <std::size_t size, class T>
std::pair<Vector<size, T>, Vector<size, T> > minmax(const Vector<size, T> &a, const Vector<size, T> &b) {
  std::pair<Vector<size, T>, Vector<size, T> > out(a, b);
  for (std::size_t i = 0; i != size; ++i)
    if (out.first[i] > out.second[i]) std::swap(out.first[i], out.second[i]);
  return out;
}

template <class T> inline T sign(const T &scalar) {
  if (scalar > T(0)) return T(1);
  if (scalar < T(0)) return T(-1);
  return T(0);
}

template <std::size_t size, class T> Vector<size, T> sign(const Vector<size, T> &a) {
  Vector<size, T> out;
  for (std::size_t i = 0; i != size; ++i) out[i] = sign(a[i]);
  return out;
}

template <std::size_t size, class T> Vector<size, T> abs(const Vector<size, T> &a) {
  Vector<size, T> out;
  for (std::size_t i = 0; i != size; ++i) out[i] = std::abs(a[i]);
  return out;
}

template <std::size_t size, class T> Vector<size, T> floor(const Vector<size, T> &a) {
  Vector<size, T> out;
  for (std::size_t i = 0; i != size; ++i) out[i] = std::floor(a[i]);
  return out;
}

template <std::size_t size, class T> Vector<size, T> ceil(const Vector<size, T> &a) {
  Vector<size, T> out;
  for (std::size_t i = 0; i != size; ++i) out[i] = std::ceil(a[i]);
  return out;
}

template <std::size_t size, class T> Vector<size, T> sqrt(const Vector<size, T> &a) {
  Vector<size, T> out;
  for (std::size_t i = 0; i != size; ++i) out[i] = T(std::sqrt(a[i]));
  return out;
}

template <std::size_t size, class T> Vector<size, T> clamp(const Vector<size, T> &value, T min, T max) {
  Vector<size, T> out;
  for (std::size_t i = 0; i != size; ++i) out[i] = clamp(value[i], min, max);
  return out;
}

template <std::size_t size, class T, class U>
inline Vector<size, T> lerp(const Vector<size, T> &a, const Vector<size, T> &b, U t) {
  return (U(1) - t) * a + t * b;
}

template <std::size_t size, class T, class U>
inline Vector<size, T> lerpInverted(const Vector<size, T> &a, const Vector<size, T> &b, const Vector<size, T> &lerp) {
  return (lerp - a) / (b - a);
}

template <std::size_t size, class T>
inline Vector<size, T> fma(const Vector<size, T> &a, const Vector<size, T> &b, const Vector<size, T> &c) {
  return a * b + c;
}
}
}
}
#endif /* GVT_CORE_MATH_FUNCTION_H */
