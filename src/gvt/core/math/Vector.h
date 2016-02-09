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
 * File:   Vector.h
 * Author: jbarbosa
 *
 * Created on March 14, 2014, 5:54 PM
 */
#ifndef GVT_CORE_MATH_VECTOR_H
#define GVT_CORE_MATH_VECTOR_H

#include <gvt/core/Debug.h>

#include <algorithm>
#include <cmath>
#include <iostream>

#ifndef GVT_ALIGNED
#if defined(__INTEL_COMPILER)
#define GVT_ALIGNED __declspec(align((16)))
#elif defined(__GNUC__)
#define GVT_ALIGNED __attribute__((aligned(16)))
#else
#define MANTA_ALIGN(size)
#endif
#endif

namespace gvt {
namespace core {
namespace math {
//==========[ Forward References ]=========================

template <class T> class Vector;
template <class T> class Vector3;
template <class T> class Vector4;
template <class T> class Matrix3;
template <class T> class Point4;
template <class T> class AffineTransformMatrix;

//==========[ Exception Classes ]==========================

class VectortorSizeMismatch {};

//==========[ class Vector ]==================================

template <class T> class Vector {
public:
  // array of elements
  T *n;
  // vector size
  int numElements;

  //---[ Constructors/Destructor ]-------------

  Vector() {
    n = NULL;
    numElements = 0;
  }

  // creates a new vector with size elements
  // if zeroElements is true the vector is initialized to zero
  Vector(int size, bool zeroElements = false);

  // copy constructor
  Vector(const Vector<T> &v);

  // destructor, simply deletes the array
  virtual ~Vector();

  //---[ Size Methods ]------------------------

  int size() const { return numElements; }

  void resize(int size, bool zeroElements = false);

  //---[ Equal Operators ]---------------------

  Vector<T> &operator=(const Vector<T> &v);
  Vector<T> &operator+=(const Vector<T> &v);
  Vector<T> &operator-=(const Vector<T> &v);
  Vector<T> &operator*=(const T d);
  Vector<T> &operator/=(const T d);

  //---[ Access Operators ]--------------------

  T &operator[](int i) { return n[i]; }

  T operator[](int i) const { return n[i]; }

  //---[ Arithmetic Operators ]----------------

  Vector<T> operator-(const Vector<T> &a);
  Vector<T> operator+(const Vector<T> &a);

  //---[ Conversion Operators ]----------------

  T *getPointer() const { return n; }

  //---[ Length Methods ]----------------------

  double length2() const;
  double length() const;

  //---[ Normalization ]-----------------------

  void normalize();

  //---[ Zero Test ]---------------------------

  bool isZero();
  void zeroElements();

  //---[ Friend Methods ]----------------------

  template <class U> friend T operator*(const Vector<T> &a, const Vector<T> &b);
  template <class U> friend Vector<T> operator-(const Vector<T> &v);
  template <class U> friend Vector<T> operator*(const Vector<T> &a, const double d);
  template <class U> friend Vector<T> operator*(const double d, const Vector<T> &a);
  template <class U> friend Vector<T> operator/(const Vector<T> &a, const double d);
  template <class U> friend Vector<T> operator^(const Vector<T> &a, const Vector<T> &b);
  template <class U> friend bool operator==(const Vector<T> &a, const Vector<T> &b);
  template <class U> friend bool operator!=(const Vector<T> &a, const Vector<T> &b);
  template <class U> friend std::ostream &operator<<(std::ostream &os, const Vector<T> &v);
  template <class U> friend std::istream &operator>>(std::istream &is, Vector<T> &v);
  template <class U> friend Vector<T> minimum(const Vector<T> &a, const Vector<T> &b);
  template <class U> friend Vector<T> maximum(const Vector<T> &a, const Vector<T> &b);
  template <class U> friend Vector<T> prod(const Vector<T> &a, const Vector<T> &b);
};

typedef Vector<int> Vectori;
typedef Vector<float> Vectorf;
typedef Vector<double> Vectord;

//==========[ class Vector2 ]=================================

template <class T> class Vector2 {
  //---[ Private Variable Declarations ]-------

public:
  union {
    struct {
      T x, y;
    };

    struct {
      T r, g;
    };
    GVT_ALIGNED T n[2];
  };

  //---[ Constructors ]------------------------

  Vector2() {
    n[0] = 0.0;
    n[1] = 0.0;
  }

  Vector2(const T x, const T y) {
    n[0] = x;
    n[1] = y;
  }

  Vector2(const Vector2<T> &v) {
    n[0] = v[0];
    n[1] = v[1];
  }

  //---[ Equal Operators ]---------------------

  Vector2<T> &operator=(const Vector2<T> &v) {
    n[0] = v[0];
    n[1] = v[1];
    return *this;
  }

  Vector2<T> &operator+=(const Vector2<T> &v) {
    n[0] += v[0];
    n[1] += v[1];
    return *this;
  }

  Vector2<T> &operator-=(const Vector2<T> &v) {
    n[0] -= v[0];
    n[1] -= v[1];
    return *this;
  }

  Vector2<T> &operator*=(const T d) {
    n[0] *= d;
    n[1] *= d;
    return *this;
  }

  Vector2<T> &operator/=(const T d) {
    n[0] /= d;
    n[1] /= d;
    return *this;
  }

  //---[ Access Operators ]--------------------

  T &operator[](int i) { return n[i]; }

  T operator[](int i) const { return n[i]; }

  //---[ Arithmetic Operators ]----------------

  Vector2<T> operator-(const Vector2<T> &a) { return Vector2<T>(n[0] - a[0], n[1] - a[1]); }

  Vector2<T> operator+(const Vector2<T> &a) { return Vector2<T>(a[0] + n[0], a[1] + n[1]); }

  //---[ Conversion Operators ]----------------

  const T *getPointer() const { return n; }

  //---[ Length Methods ]----------------------

  double length2() const { return n[0] * n[0] + n[1] * n[1]; }

  double length() const { return sqrt(length2()); }

  //---[ Normalization ]-----------------------

  void normalize() {
    double len = 1. / length();
    n[0] *= len;
    n[1] *= len;
  }

  //---[ Zero Test ]---------------------------
  // clang-format off
  bool isZero() { return ((n[0] == 0 && n[1] == 0) ? true : false); }
  // clang-format on

  void zeroElements() { memset(n, 0, sizeof(T) * 2); }
};

typedef Vector2<int> Vector2i;
typedef Vector2<float> Vector2f;
typedef Vector2<double> Vector2d;

//==========[ class Vector3 ]=================================

template <class T> class Vector3 {
  //---[ Private Variable Declarations ]-------
public:
  // x, y, z
  union {

    struct {
      T x, y, z;
    };

    struct {
      T r, g, b;
    };

    GVT_ALIGNED T n[3];
  };

  //---[ Constructors ]------------------------

  Vector3() {
    n[0] = 0.0;
    n[1] = 0.0;
    n[2] = 0.0;
  }

  Vector3(const T x, const T y, const T z) {
    n[0] = x;
    n[1] = y;
    n[2] = z;
  }

  Vector3(const T *x) {
    n[0] = x[0];
    n[1] = x[1];
    n[2] = x[2];
  }

  Vector3(const unsigned char *buf) {
    T *x = (T *)buf;
    n[0] = x[0];
    n[1] = x[1];
    n[2] = x[2];
  }

  inline int packedSize() { return 3 * sizeof(T); }

  inline int pack(unsigned char *buffer) {
    T *x = (T *)buffer;
    x[0] = n[0];
    x[1] = n[1];
    x[2] = n[2];
    return 3 * sizeof(T);
  }

  Vector3(const Vector3<T> &v) {
    n[0] = v[0];
    n[1] = v[1];
    n[2] = v[2];
  }

  Vector3(int) {
    n[0] = 0.0;
    n[1] = 0.0;
    n[2] = 0.0;
  }

  Vector3(const Vector4<T> &v) {
    n[0] = v[0];
    n[1] = v[1];
    n[2] = v[2];
  }

  //---[ Equal Operators ]---------------------

  Vector3<T> &operator=(const Vector3<T> &v) {
    n[0] = v[0];
    n[1] = v[1];
    n[2] = v[2];
    return *this;
  }

  Vector3<T> &operator=(const Vector4<T> &v) {
    n[0] = v[0];
    n[1] = v[1];
    n[2] = v[2];
    return *this;
  }

  Vector3<T> &operator+=(const Vector3<T> &v) {
    n[0] += v[0];
    n[1] += v[1];
    n[2] += v[2];
    return *this;
  }

  Vector3<T> &operator-=(const Vector3<T> &v) {
    n[0] -= v[0];
    n[1] -= v[1];
    n[2] -= v[2];
    return *this;
  }

  Vector3<T> &operator*=(const T d) {
    n[0] *= d;
    n[1] *= d;
    n[2] *= d;
    return *this;
  }

  Vector3<T> &operator/=(const T d) {
    n[0] /= d;
    n[1] /= d;
    n[2] /= d;
    return *this;
  }

  //---[ Access Operators ]--------------------

  T &operator[](int i) { return n[i]; }

  T operator[](int i) const { return n[i]; }

  //---[ Arithmetic Operators ]----------------

  Vector3<T> operator-(const Vector3<T> &a) const { return Vector3<T>(n[0] - a[0], n[1] - a[1], n[2] - a[2]); }

  Vector3<T> operator+(const Vector3<T> &a) const { return Vector3<T>(a[0] + n[0], a[1] + n[1], a[2] + n[2]); }

  //---[ Conversion Operators ]----------------

  const T *getPointer() const { return n; }

  //---[ Length Methods ]----------------------

  double length2() const { return n[0] * n[0] + n[1] * n[1] + n[2] * n[2]; }

  double length() const { return sqrt(length2()); }

  //---[ Normalization ]-----------------------

  Vector3<T> normalize() {
    double len = 1. / length();
    n[0] *= len;
    n[1] *= len;
    n[2] *= len;
    return (*this);
  }

  void clamp() {
    int i;
    for (i = 0; i < 3; i++) {
      if (n[i] < 0) n[i] = 0.0;
      if (n[i] > 1) n[i] = 1.0;
    }
  }

  //---[ Zero Test ]---------------------------
  // clang-format off
  bool isZero() {
    return ((n[0] == 0 && n[1] == 0 && n[2] == 0) ? true : false);
  };
  // clang-format on

  void zeroElements() { memset(n, 0, sizeof(T) * 3); }

  //---[ OpenGL Methods ]----------------------

  void glTranslate() { glTranslated(n[0], n[1], n[2]); }

  void glColor() { glColor3d(n[0], n[1], n[2]); }

  void glVertex() { glVertex3d(n[0], n[1], n[2]); }

  void glNormal() { glNormal3d(n[0], n[1], n[2]); }

  //---[ Friend Methods ]----------------------

  template <class U> friend T operator*(const Vector3<T> &a, const Vector4<T> &b);
  template <class U> friend T operator*(const Vector4<T> &b, const Vector3<T> &a);
  template <class U> friend Vector3<T> operator*(const Vector3<T> &a, const double d);
  template <class U> friend Vector3<T> operator*(const double d, const Vector3<T> &a);
  template <class U> friend Vector3<T> operator*(const Vector3<T> &v, AffineTransformMatrix<T> &a);
  template <class U> friend T operator*(const Vector3<T> &a, const Vector3<T> &b);
  template <class U> friend Vector3<T> operator*(const Matrix3<T> &a, const Vector3<T> &v);
  template <class U> friend Vector3<T> operator*(const Vector3<T> &v, const Matrix3<T> &a);
  template <class U> friend Vector3<T> operator*(const AffineTransformMatrix<T> &a, const Vector3<T> &v);
  template <class U> friend Vector3<T> operator/(const Vector3<T> &a, const double d);
  template <class U> friend Vector3<T> operator^(const Vector3<T> &a, const Vector3<T> &b);
  template <class U> friend bool operator==(const Vector3<T> &a, const Vector3<T> &b);
  template <class U> friend bool operator!=(const Vector3<T> &a, const Vector3<T> &b);
  template <class U> friend std::ostream &operator<<(std::ostream &os, const Vector3<T> &v);
  template <class U> friend std::istream &operator>>(std::istream &is, Vector3<T> &v);
  template <class U> friend Vector3<T> minimum(const Vector3<T> &a, const Vector3<T> &b);
  template <class U> friend Vector3<T> maximum(const Vector3<T> &a, const Vector3<T> &b);
  template <class U> friend Vector3<T> prod(const Vector3<T> &a, const Vector3<T> &b);
};

typedef Vector3<int> Vector3i;
typedef Vector3<float> Vector3f;
typedef Vector3<double> Vector3d;

//==========[ class Vector4 ]=================================

template <class T> class Point4;

template <class T> class Vector4 {
  //---[ Private Variable Declarations ]-------
public:
  union {
    struct {
      T x, y, z, w;
    };

    struct {
      T r, g, b, a;
    };

    GVT_ALIGNED T n[4];
  };

  //---[ Constructors ]------------------------

  Vector4() {
    n[0] = 0.0;
    n[1] = 0.0;
    n[2] = 0.0;
    n[3] = 0.0;
  }

  Vector4(const T x, const T y, const T z, const T w) {
    n[0] = x;
    n[1] = y;
    n[2] = z;
    n[3] = w;
  }

  Vector4(const Vector4 &v) {
    n[0] = v[0];
    n[1] = v[1];
    n[2] = v[2];
    n[3] = v[3];
  }

  Vector4(const Point4<T> &v) {
    n[0] = v[0];
    n[1] = v[1];
    n[2] = v[2];
    n[3] = v[3];
  }

  Vector4(Point4<T> &v) {
    n[0] = v[0];
    n[1] = v[1];
    n[2] = v[2];
    n[3] = v[3];
  }

  Vector4(const Vector3<T> &v) {
    n[0] = v[0];
    n[1] = v[1];
    n[2] = v[2];
    n[3] = 0.f;
  }

  Vector4(const unsigned char *buf) {
    T *x = (T *)buf;
    n[0] = x[0];
    n[1] = x[1];
    n[2] = x[2];
    n[3] = x[3];
  }

  inline int packedSize() { return 4 * sizeof(T); }

  inline int pack(unsigned char *buffer) {
    T *x = (T *)buffer;
    x[0] = n[0];
    x[1] = n[1];
    x[2] = n[2];
    x[3] = n[3];
    return 4 * sizeof(T);
  }

  //---[ Equal Operators ]---------------------

  Vector4<T> &operator=(const Vector4<T> &v) {
    n[0] = v[0];
    n[1] = v[1];
    n[2] = v[2];
    n[3] = v[3];
    return *this;
  }

  Vector4<T> &operator=(const Vector3<T> &v) {
    n[0] = v[0];
    n[1] = v[1];
    n[2] = v[2];
    n[3] = 0.f;
    return *this;
  }

  Vector4<T> &operator+=(const Vector4<T> &v) {
    n[0] += v[0];
    n[1] += v[1];
    n[2] += v[2];
    n[3] += v[3];
    return *this;
  }

  Vector4<T> &operator-=(const Vector4<T> &v) {
    n[0] -= v[0];
    n[1] -= v[1];
    n[2] -= v[2];
    n[3] -= v[3];
    return *this;
  }

  Vector4<T> &operator*=(const T d) {
    n[0] *= d;
    n[1] *= d;
    n[2] *= d;
    n[3] *= d;
    return *this;
  }

  Vector4<T> &operator/=(const T d) {
    n[0] /= d;
    n[1] /= d;
    n[2] /= d;
    n[3] /= d;
    return *this;
  }

  //---[ Access Operators ]--------------------

  T &operator[](int i) { return n[i]; }

  T operator[](int i) const { return n[i]; }

  //---[ Arithmetic Operators ]----------------

  Vector4<T> operator-(const Vector4<T> &a) const {
    return Vector4<T>(n[0] - a[0], n[1] - a[1], n[2] - a[2], n[3] - a[3]);
  }

  Vector4<T> operator+(const Vector4<T> &a) { return Vector4<T>(a[0] + n[0], a[1] + n[1], a[2] + n[2], a[3] + n[3]); }

  //---[ Length Methods ]----------------------

  double length2() const { return n[0] * n[0] + n[1] * n[1] + n[2] * n[2] + n[3] * n[3]; }

  double length() const { return sqrt(length2()); }

  //---[ Zero Test ]---------------------------

  bool isZero() const { return n[0] == 0 && n[1] == 0 && n[2] == 0 && n[3] == 0; }

  void zeroElements() { memset(n, 0, 4 * sizeof(T)); }

  //---[ Normalization ]-----------------------

  Vector4<T> normalize() {
    double len = 1. / length();
    n[0] *= len;
    n[1] *= len;
    n[2] *= len;
    n[3] *= len;
    return (*this);
  }

  //---[ Friend Methods ]----------------------
  template <class U> friend T operator*(const Vector3<T> &a, const Vector4<T> &b);
  template <class U> friend T operator*(const Vector4<T> &b, const Vector3<T> &a);
  //	template <class U> friend Vector4<T> operator -( const Vector4<T>& v );
  template <class U> friend Vector4<T> operator*(const Vector4<T> &a, const double d);
  template <class U> friend Vector4<T> operator*(const double d, const Vector4<T> &a);
  template <class U> friend T operator*(const Vector4<T> &a, const Vector4<T> &b);
  template <class U> friend Vector4<T> operator*(const AffineTransformMatrix<T> &a, const Vector4<T> &v);
  //	template <class U> friend Vector4<T> operator *( const Vector4<T>& v, const Matrix4<T>& a );
  template <class U> friend Vector4<T> operator/(const Vector4<T> &a, const double d);
  //	template <class U> friend Vector4<T> operator ^( const Vector4<T>& a, const Vector4<T>& b );
  template <class U> friend bool operator==(const Vector4<T> &a, const Vector4<T> &b);
  template <class U> friend bool operator!=(const Vector4<T> &a, const Vector4<T> &b);
  template <class U> friend std::ostream &operator<<(std::ostream &os, const Vector4<T> &v);
  template <class U> friend std::istream &operator>>(std::istream &is, Vector4<T> &v);
  template <class U> friend Vector4<T> minimum(const Vector4<T> &a, const Vector4<T> &b);
  template <class U> friend Vector4<T> maximum(const Vector4<T> &a, const Vector4<T> &b);
  template <class U> friend Vector4<T> prod(const Vector4<T> &a, const Vector4<T> &b);
};

typedef Vector4<int> Vector4i;
typedef Vector4<float> Vector4f;
typedef Vector4<double> Vector4d;

template <class T> class Point4 : public Vector4<T> {
public:
  //---[ Constructors ]------------------------
  Point4() {
    this->n[0] = 0.0;
    this->n[1] = 0.0;
    this->n[2] = 0.0;
    this->n[3] = 1.0;
  }

  Point4(const T x, const T y, const T z, const T w = T(1)) {
    this->n[0] = x;
    this->n[1] = y;
    this->n[2] = z;
    this->n[3] = w;
  }

  Point4(const Vector4<T> &v) {
    this->n[0] = v[0];
    this->n[1] = v[1];
    this->n[2] = v[2];
    this->n[3] = v[3];
  }

  Point4(Vector4<T> &v) {
    this->n[0] = v[0];
    this->n[1] = v[1];
    this->n[2] = v[2];
    this->n[3] = v[3];
  }

  Point4(const Point4 &v) {
    this->n[0] = v[0];
    this->n[1] = v[1];
    this->n[2] = v[2];
    this->n[3] = v[3];
  }

  Point4(const unsigned char *buf) {
    T *v = (T *)buf;
    this->n[0] = v[0];
    this->n[1] = v[1];
    this->n[2] = v[2];
    this->n[3] = v[3];
  }

  double length2() const { return 0.f; }

  double length() const { return 0.f; }

  template <class U> friend std::ostream &operator<<(std::ostream &os, const Point4<T> &m);
  template <class U> friend std::istream &operator>>(std::istream &is, Point4<T> &m);
};

typedef Point4<int> Point4i;
typedef Point4<float> Point4f;
typedef Point4<double> Point4d;

//==========[ Vector Methods ]================================

template <class T> Vector<T>::Vector(int size, bool zeroElements) {
  numElements = size;
  n = new T[size];

  if (!zeroElements) return;

  for (int i = 0; i < size; i++) n[i] = 0.0;
}

template <class T> Vector<T>::Vector(const Vector<T> &v) {
  numElements = v.numElements;
  n = new T[numElements];

  memcpy(n, v.n, numElements * sizeof(T));
}

template <class T> Vector<T>::~Vector() { delete[] n; }

template <class T> void Vector<T>::resize(int size, bool zeroElements) {
  if (numElements != size) {
    numElements = size;
    delete[] n;

    n = new T[size];
  }

  if (zeroElements) memset(n, 0, numElements * sizeof(T));
}

template <class T> void Vector<T>::zeroElements() { memset(n, 0, numElements * sizeof(T)); }

template <class T> Vector<T> &Vector<T>::operator=(const Vector<T> &v) {
  GVT_DEBUG_CODE(DBG_ALWAYS, if (v.numElements != numElements) throw VectortorSizeMismatch());

  for (int i = 0; i < numElements; i++) n[i] = v[i];

  return *this;
}

template <class T> Vector<T> &Vector<T>::operator+=(const Vector<T> &v) {
  GVT_DEBUG_CODE(DBG_ALWAYS, if (v.numElements != numElements) throw VectortorSizeMismatch());

  for (int i = 0; i < numElements; i++) n[i] += v[i];

  return *this;
}

template <class T> Vector<T> &Vector<T>::operator-=(const Vector<T> &v) {
  GVT_DEBUG_CODE(DBG_ALWAYS, if (v.numElements != numElements) throw VectortorSizeMismatch());

  for (int i = 0; i < numElements; i++) n[i] -= v[i];

  return *this;
}

template <class T> Vector<T> &Vector<T>::operator*=(const T d) {
  for (int i = 0; i < numElements; i++) n[i] *= d;

  return *this;
}

template <class T> Vector<T> &Vector<T>::operator/=(const T d) {
  for (int i = 0; i < numElements; i++) n[i] /= d;

  return *this;
}

template <class T> Vector<T> Vector<T>::operator-(const Vector<T> &v) {
  GVT_DEBUG_CODE(DBG_ALWAYS, if (v.numElements != numElements) throw VectortorSizeMismatch());

  Vector<T> result(numElements, false);

  for (int i = 0; i < numElements; i++) result[i] = n[i] - v[i];

  return result;
}

template <class T> Vector<T> Vector<T>::operator+(const Vector<T> &v) {
  GVT_DEBUG_CODE(DBG_ALWAYS, if (v.numElements != numElements) throw VectortorSizeMismatch());

  Vector<T> result(numElements, false);

  for (int i = 0; i < numElements; i++) result[i] = v[i] + n[i];

  return result;
}

template <class T> double Vector<T>::length2() const {
  double result = 0.0;

  for (int i = 0; i < numElements; i++) result += n[i] * n[i];

  return result;
}

template <class T> double Vector<T>::length() const { return sqrt(length2()); }

template <class T> void Vector<T>::normalize() {
  double len = 1. / length();

  for (int i = 0; i < numElements; i++) n[i] *= len;
}

template <class T> bool Vector<T>::isZero() {
  for (int i = 0; i < numElements; i++)
    if (n[i] != 0) return false;

  return true;
}

template <class T> Vector<T> minimum(const Vector<T> &a, const Vector<T> &b) {
  GVT_DEBUG_CODE(DBG_ALWAYS, if (a.numElements != b.numElements) throw VectortorSizeMismatch());

  gvt::core::math::Vector<T> result(a.numElements, false);

  for (int i = 0; i < a.numElements; i++) result[i] = min(a[i], b[i]);

  return result;
}

template <class T> Vector<T> maximum(const Vector<T> &a, const Vector<T> &b) {
  GVT_DEBUG_CODE(DBG_ALWAYS, if (a.numElements != b.numElements) throw VectortorSizeMismatch());

  Vector<T> result(a.numElements, false);

  for (int i = 0; i < a.numElements; i++) result[i] = max(a[i], b[i]);

  return result;
}

template <class T> Vector<T> prod(const Vector<T> &a, const Vector<T> &b) {
  GVT_DEBUG_CODE(DBG_ALWAYS, if (a.numElements != b.numElements) throw VectortorSizeMismatch());

  Vector<T> result(a.numElements, false);

  for (int i = 0; i < a.numElements; i++) result[i] = a[i] * b[i];

  return result;
}

template <class T> inline Vector3<T> minimum(const Vector3<T> &a, const Vector3<T> &b) {
  return gvt::core::math::Vector3<T>(min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]));
}

template <class T> inline Vector3<T> maximum(const Vector3<T> &a, const Vector3<T> &b) {
  return gvt::core::math::Vector3<T>(max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]));
}

template <class T> inline Vector3<T> prod(const Vector3<T> &a, const Vector3<T> &b) {
  return gvt::core::math::Vector3<T>(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}

template <class T> inline Vector4<T> minimum(const Vector4<T> &a, const Vector4<T> &b) {
  return gvt::core::math::Vector4<T>(min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]));
}

template <class T> inline Vector4<T> maximum(const Vector4<T> &a, const Vector4<T> &b) {
  return gvt::core::math::Vector4<T>(max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]));
}

template <class T> inline Vector4<T> prod(const Vector4<T> &a, const Vector4<T> &b) {
  return gvt::core::math::Vector4<T>(a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]);
}

template <class T> Vector3<T> vector4to3(Vector4<T> &v) { return Vector3<T>(v[0], v[1], v[2]); }
#include "VectorOperators.inl"
}
}
}

#endif // GVT_CORE_MATH_VECTOR_H
