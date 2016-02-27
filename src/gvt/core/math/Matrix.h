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
 * File:   Matrix.h
 * Author: jbarbosa
 *
 * Created on March 14, 2014, 5:54 PM
 */

#ifndef GVT_CORE_MATH_MATRIX_H
#define GVT_CORE_MATH_MATRIX_H

#include <iostream>
#include <cmath>
#include <string.h>

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

//==========[ Forward References ]=============================================

template <class T> class Vector;
template <class T> class Vector3;
template <class T> class Vector4;
template <class T> class Matrix3;
template <class T> class AffineTransformMatrix;

//==========[ class Matrix3 ]=====================================================
template <class T> class Matrix3 {
  //---[ Private Variable Declarations ]-----------------
public:
  // matrix elements in row major order
  T n[9];

  //---[ Constructors ]----------------------------------

  Matrix3() {
    memset(n, 0, 9 * sizeof(T));
    n[0] = 1;
    n[4] = 1;
    n[8] = 1;
  }

  Matrix3(T n0, T n1, T n2, T n3, T n4, T n5, T n6, T n7, T n8) {
    n[0] = n0;
    n[1] = n1;
    n[2] = n2;
    n[3] = n3;
    n[4] = n4;
    n[5] = n5;
    n[6] = n6;
    n[7] = n7;
    n[8] = n8;
  }

  Matrix3(const Matrix3<T> &m) { memcpy(n, m.n, 9 * sizeof(T)); }

  //---[ Equal Operators ]-------------------------------

  Matrix3<T> &operator=(const Matrix3<T> &m) {
    memcpy(n, m.n, 9 * sizeof(T));
    return *this;
  }

  Matrix3<T> &operator+=(const Matrix3<T> &m) {
    n[0] += m.n[0];
    n[1] += m.n[1];
    n[2] += m.n[2];
    n[3] += m.n[3];
    n[4] += m.n[4];
    n[5] += m.n[5];
    n[6] += m.n[6];
    n[7] += m.n[7];
    n[8] += m.n[8];
    return *this;
  }

  Matrix3<T> &operator-=(const Matrix3<T> &m) {
    n[0] -= m.n[0];
    n[1] -= m.n[1];
    n[2] -= m.n[2];
    n[3] -= m.n[3];
    n[4] -= m.n[4];
    n[5] -= m.n[5];
    n[6] -= m.n[6];
    n[7] -= m.n[7];
    n[8] -= m.n[8];
    return *this;
  }

  Matrix3<T> &operator*=(const T d) {
    n[0] *= d;
    n[1] *= d;
    n[2] *= d;
    n[3] *= d;
    n[4] *= d;
    n[5] *= d;
    n[6] *= d;
    n[7] *= d;
    n[8] *= d;
    return *this;
  }

  Matrix3<T> &operator/=(const T d) {
    n[0] /= d;
    n[1] /= d;
    n[2] /= d;
    n[3] /= d;
    n[4] /= d;
    n[5] /= d;
    n[6] /= d;
    n[7] /= d;
    n[8] /= d;
    return *this;
  }

  //---[ Access Operators ]------------------------------

  T *operator[](int i) { return &n[i * 3]; }
  const T *operator[](int i) const { return &n[i * 3]; }

  //---[ Ordering Methods ]------------------------------

  Matrix3<T> transpose() const { return Matrix3<T>(n[0], n[3], n[6], n[1], n[4], n[7], n[2], n[5], n[8]); }

  double trace() const { return n[0] + n[4] + n[8]; }

  //---[ GL Matrixrix ]-------------------------------------

  void getGLMatrixrix(T *mat) const {
    mat[0] = n[0];
    mat[1] = n[3];
    mat[2] = n[6];
    mat[3] = n[1];
    mat[4] = n[4];
    mat[5] = n[7];
    mat[6] = n[2];
    mat[7] = n[5];
    mat[8] = n[8];
  }

  //---[ Transformation Matrixrices ]-----------------------

  static Matrix3<T> createRotation(T angle, float x, float y);
  static Matrix3<T> createTranslation(T x, T y);
  static Matrix3<T> createScale(T sx, T sy);
  static Matrix3<T> createShear(T shx, T shy);

  //---[ Inversion ]-------------------------------------

  Matrix3<T> inverse() const // Gauss-Jordan elimination with partial pivoting
  {
    Matrix3<T> a(*this); // As a evolves from original mat into identity
    Matrix3<T> b;        // b evolves from identity into inverse(a)
    int i, j, i1;

    // Loop over cols of a from left to right, eliminating above and below diag
    for (j = 0; j < 3; j++) // Find largest pivot in column j among rows j..2
    {
      i1 = j; // Row with largest pivot candidate
      for (i = j + 1; i < 3; i++) {
        if (std::abs(a[i][j]) > std::abs(a[i1][j])) i1 = i;

        // Swap rows i1 and j in a and b to put pivot on diagonal
        for (i = 0; i < 3; i++) {
          std::swap(a[i1][i], a[j][i]);
          std::swap(b[i1][i], b[j][i]);
        }

        double scale = a[j][j];
        for (i = 0; i < 3; i++) {
          b[j][i] /= scale;
          a[j][i] /= scale;
        }

        // Eliminate off-diagonal elems in col j of a, doing identical ops to b
        for (i = 0; i < 3; i++) {
          if (i != j) {
            scale = a[i][j];
            for (i1 = 0; i1 < 3; i1++) {
              b[i][i1] -= scale * b[j][i1];
              a[i][i1] -= scale * a[j][i1];
            }
          }
        }
      }
    }
    return b;
  }

  //---[ Friend Methods ]--------------------------------

  template <class U> friend Matrix3<T> operator-(const Matrix3<T> &a);
  template <class U> friend Matrix3<T> operator+(const Matrix3<T> &a, const Matrix3<T> &b);
  template <class U> friend Matrix3<T> operator-(const Matrix3<T> &a, const Matrix3<T> &b);
  template <class U> friend Matrix3<T> operator*(const Matrix3<T> &a, const Matrix3<T> &b);
  template <class U> friend Matrix3<T> operator*(const Matrix3<T> &a, const double d);
  template <class U> friend Matrix3<T> operator*(const double d, const Matrix3<T> &a);
  template <class U> friend Matrix3<T> operator/(const Matrix3<T> &a, const double d);
  template <class U> friend bool operator==(const Matrix3<T> &a, const Matrix3<T> &b);
  template <class U> friend bool operator!=(const Matrix3<T> &a, const Matrix3<T> &b);
  template <class U> friend std::ostream &operator<<(std::ostream &os, const Matrix3<T> &m);
  template <class U> friend std::istream &operator>>(std::istream &is, Matrix3<T> &m);
};

typedef Matrix3<int> Matrix3i;
typedef Matrix3<float> Matrix3f;
typedef Matrix3<double> Matrix3d;

//==========[ class Matrix4 ]=====================================================

template <class T> class AffineTransformMatrix {
  //---[ Private Variable Declarations ]-----------------
public:
  // matrix elements in row-major order
  T n[16];

  bool isZero() {
    return n[0] == 0 && n[1] == 0 && n[2] == 0 && n[3] == 0 && n[4] == 0 && n[5] == 0 && n[6] == 0 && n[7] == 0 &&
           n[8] == 0 && n[9] == 0 && n[10] == 0 && n[11] == 0 && n[12] == 0 && n[13] == 0 && n[14] == 0 && n[15] == 0;
  }

  //---[ Constructors ]----------------------------------

  AffineTransformMatrix(bool identity = true) {
    memset(n, 0, 16 * sizeof(T));
    if (identity) {
      n[0] = 1;
      n[5] = 1;
      n[10] = 1;
      n[15] = 1;
    }
  }

  AffineTransformMatrix(T n0, T n1, T n2, T n3, T n4, T n5, T n6, T n7, T n8, T n9, T n10, T n11, T n12, T n13, T n14,
                        T n15) {
    n[0] = n0;
    n[1] = n1;
    n[2] = n2;
    n[3] = n3;
    n[4] = n4;
    n[5] = n5;
    n[6] = n6;
    n[7] = n7;
    n[8] = n8;
    n[9] = n9;
    n[10] = n10;
    n[11] = n11;
    n[12] = n12;
    n[13] = n13;
    n[14] = n14;
    n[15] = n15;
  }

  AffineTransformMatrix(const AffineTransformMatrix<T> &m) { memcpy(n, m.n, 16 * sizeof(T)); }

  AffineTransformMatrix(const Vector4<T> &v0, const Vector4<T> &v1, const Vector4<T> &v2, const Vector4<T> &v3) {
    n[0] = v0[0];
    n[1] = v0[1];
    n[2] = v0[2];
    n[3] = v0[3];
    n[4] = v1[0];
    n[5] = v1[1];
    n[6] = v1[2];
    n[7] = v1[3];
    n[8] = v2[0];
    n[9] = v2[1];
    n[10] = v2[2];
    n[11] = v2[3];
    n[12] = v3[0];
    n[13] = v3[1];
    n[14] = v3[2];
    n[15] = v3[3];
  }

  //---[ Equal Operators ]-------------------------------

  AffineTransformMatrix<T> &operator=(const AffineTransformMatrix<T> &m) {
    memcpy(n, m.n, 16 * sizeof(T));
    return *this;
  }

  AffineTransformMatrix<T> &operator+=(const AffineTransformMatrix<T> &m) {
    n[0] += m.n[0];
    n[1] += m.n[1];
    n[2] += m.n[2];
    n[3] += m.n[3];
    n[4] += m.n[4];
    n[5] += m.n[5];
    n[6] += m.n[6];
    n[7] += m.n[7];
    n[8] += m.n[8];
    n[9] += m.n[9];
    n[10] += m.n[10];
    n[11] += m.n[11];
    n[12] += m.n[12];
    n[13] += m.n[13];
    n[14] += m.n[14];
    n[15] += m.n[15];
    return *this;
  }

  AffineTransformMatrix<T> &operator-=(const AffineTransformMatrix<T> &m) {
    n[0] -= m.n[0];
    n[1] -= m.n[1];
    n[2] -= m.n[2];
    n[3] -= m.n[3];
    n[4] -= m.n[4];
    n[5] -= m.n[5];
    n[6] -= m.n[6];
    n[7] -= m.n[7];
    n[8] -= m.n[8];
    n[9] -= m.n[9];
    n[10] -= m.n[10];
    n[11] -= m.n[11];
    n[12] -= m.n[12];
    n[13] -= m.n[13];
    n[14] -= m.n[14];
    n[15] -= m.n[15];
    return *this;
  }

  AffineTransformMatrix<T> &operator*=(const T d) {
    n[0] *= d;
    n[1] *= d;
    n[2] *= d;
    n[3] *= d;
    n[4] *= d;
    n[5] *= d;
    n[6] *= d;
    n[7] *= d;
    n[8] *= d;
    n[9] *= d;
    n[10] *= d;
    n[11] *= d;
    n[12] *= d;
    n[13] *= d;
    n[14] *= d;
    n[15] *= d;
    return *this;
  }

  AffineTransformMatrix<T> &operator/=(const T d) {
    n[0] /= d;
    n[1] /= d;
    n[2] /= d;
    n[3] /= d;
    n[4] /= d;
    n[5] /= d;
    n[6] /= d;
    n[7] /= d;
    n[8] /= d;
    n[9] /= d;
    n[10] /= d;
    n[11] /= d;
    n[12] /= d;
    n[13] /= d;
    n[14] /= d;
    n[15] /= d;
    return *this;
  }

  //---[ Access Operators ]------------------------------

  T *operator[](int i) { return &n[i * 4]; }

  const T *operator[](int i) const { return &n[i * 4]; }

  //---[ Ordering Methods ]------------------------------

  AffineTransformMatrix<T> transpose() const {
    return AffineTransformMatrix<T>(n[0], n[4], n[8], n[12], n[1], n[5], n[9], n[13], n[2], n[6], n[10], n[14], n[3],
                                    n[7], n[11], n[15]);
  }

  double trace() const { return n[0] + n[5] + n[10] + n[15]; }

  //---[ GL Matrixrix ]-------------------------------------

  void getGLMatrixrix(T *mat) const {
    mat[0] = n[0];
    mat[1] = n[4];
    mat[2] = n[8];
    mat[3] = n[12];
    mat[4] = n[1];
    mat[5] = n[5];
    mat[6] = n[9];
    mat[7] = n[13];
    mat[8] = n[2];
    mat[9] = n[6];
    mat[10] = n[10];
    mat[11] = n[14];
    mat[12] = n[3];
    mat[13] = n[7];
    mat[14] = n[11];
    mat[15] = n[15];
  }

  void getGLMatrixrixF(float *mat) {
    mat[0] = n[0];
    mat[1] = n[4];
    mat[2] = n[8];
    mat[3] = n[12];
    mat[4] = n[1];
    mat[5] = n[5];
    mat[6] = n[9];
    mat[7] = n[13];
    mat[8] = n[2];
    mat[9] = n[6];
    mat[10] = n[10];
    mat[11] = n[14];
    mat[12] = n[3];
    mat[13] = n[7];
    mat[14] = n[11];
    mat[15] = n[15];
  }

  //---[ Transformation Matrixrices ]-----------------------

  static AffineTransformMatrix<T> createRotation(T angle, float x, float y, float z);
  static AffineTransformMatrix<T> createTranslation(T x, T y, T z);
  static AffineTransformMatrix<T> createScale(T sx, T sy, T sz);
  static AffineTransformMatrix<T> createShear(T shx, T shy, T shz);

  //---[ Conversion ]------------------------------------

  Matrix3<T> upper33() { return Matrix3<T>(n[0], n[1], n[2], n[4], n[5], n[6], n[8], n[9], n[10]); }

  //---[ Inversion ]-------------------------------------

  AffineTransformMatrix<T> inverse() const // Gauss-Jordan elimination with partial pivoting
  {
    AffineTransformMatrix<T> a(*this); // As a evolves from original mat into identity
    AffineTransformMatrix<T> b;        // b evolves from identity into inverse(a)
    int i, j, i1;

    // Loop over cols of a from left to right, eliminating above and below diag
    for (j = 0; j < 4; j++) // Find largest pivot in column j among rows j..2
    {
      i1 = j; // Row with largest pivot candidate
      for (i = j + 1; i < 4; i++) {
        if (std::abs(a[i][j]) > std::abs(a[i1][j])) i1 = i;
      }

      // Swap rows i1 and j in a and b to put pivot on diagonal
      for (i = 0; i < 4; i++) {
        std::swap(a[i1][i], a[j][i]);
        std::swap(b[i1][i], b[j][i]);
      }

      double scale = a[j][j];
      for (i = 0; i < 4; i++) {
        b[j][i] /= scale;
        a[j][i] /= scale;
      }

      // Eliminate off-diagonal elems in col j of a, doing identical ops to b
      for (i = 0; i < 4; i++) {
        if (i != j) {
          scale = a[i][j];
          for (i1 = 0; i1 < 4; i1++) {
            b[i][i1] -= scale * b[j][i1];
            a[i][i1] -= scale * a[j][i1];
          }
        }
      }
    }
    return b;
  }

  //---[ Friend Methods ]--------------------------------

  template <class U> friend AffineTransformMatrix<T> operator-(const AffineTransformMatrix<T> &a);
  template <class U>
  friend AffineTransformMatrix<T> operator+(const AffineTransformMatrix<T> &a, const AffineTransformMatrix<T> &b);
  template <class U>
  friend AffineTransformMatrix<T> operator-(const AffineTransformMatrix<T> &a, const AffineTransformMatrix<T> &b);
  template <class U>
  friend AffineTransformMatrix<T> operator*(const AffineTransformMatrix<T> &a, const AffineTransformMatrix<T> &b);
  template <class U> friend AffineTransformMatrix<T> operator*(const AffineTransformMatrix<T> &a, const double d);
  template <class U> friend AffineTransformMatrix<T> operator*(const double d, const AffineTransformMatrix<T> &a);
  template <class U> friend Vector3<T> operator*(const AffineTransformMatrix<T> &a, const Vector3<T> &b);
  template <class U> friend AffineTransformMatrix<T> operator/(const AffineTransformMatrix<T> &a, const double d);
  template <class U> friend bool operator==(const AffineTransformMatrix<T> &a, const AffineTransformMatrix<T> &b);
  template <class U> friend bool operator!=(const AffineTransformMatrix<T> &a, const AffineTransformMatrix<T> &b);
  template <class U> friend std::ostream &operator<<(std::ostream &os, const AffineTransformMatrix<T> &m);
  template <class U> friend std::istream &operator>>(std::istream &is, AffineTransformMatrix<T> &m);
};

typedef AffineTransformMatrix<int> Matrix4i;
typedef AffineTransformMatrix<float> Matrix4f;
typedef AffineTransformMatrix<double> Matrix4d;

//==========[ Inline Method Definitions (Matrixrix) ]=============================

template <class T> inline Matrix3<T> Matrix3<T>::createRotation(T angle, float x, float y) {
  Matrix3<T> rot;
  GVT_DEBUG(DBG_ALWAYS, "unimplemented matrix command createRotation(angle,x,y)");
  return rot;
}

template <class T> inline Matrix3<T> Matrix3<T>::createTranslation(T x, T y) {
  Matrix3<T> trans;
  GVT_DEBUG(DBG_ALWAYS, "unimplemented matrix command createTranslation(x,y)");
  return trans;
}

template <class T> inline Matrix3<T> Matrix3<T>::createScale(T sx, T sy) {
  Matrix3<T> scale;
  GVT_DEBUG(DBG_ALWAYS, "unimplemented matrix command createScale(sx,sy)");
  return scale;
}

template <class T> inline Matrix3<T> Matrix3<T>::createShear(T shx, T shy) {
  Matrix3<T> shear;
  GVT_DEBUG(DBG_ALWAYS, "unimplemented matrix command createShear(shx,shy)");
  return shear;
}

template <class T> inline Matrix3<T> operator-(const Matrix3<T> &a) {
  return Matrix3<T>(-a.n[0], -a.n[1], -a.n[2], -a.n[3], -a.n[4], -a.n[5], -a.n[6], -a.n[7], -a.n[8]);
}

template <class T> inline Matrix3<T> operator+(const Matrix3<T> &a, const Matrix3<T> &b) {
  return Matrix3<T>(a.n[0] + b.n[0], a.n[1] + b.n[1], a.n[2] + b.n[2], a.n[3] + b.n[3], a.n[4] + b.n[4],
                    a.n[5] + b.n[5], a.n[6] + b.n[6], a.n[7] + b.n[7], a.n[8] + b.n[8]);
}

template <class T> inline Matrix3<T> operator-(const Matrix3<T> &a, const Matrix3<T> &b) {
  return Matrix3<T>(a.n[0] - b.n[0], a.n[1] - b.n[1], a.n[2] - b.n[2], a.n[3] - b.n[3], a.n[4] - b.n[4],
                    a.n[5] - b.n[5], a.n[6] - b.n[6], a.n[7] - b.n[7], a.n[8] - b.n[8]);
}

template <class T> inline Matrix3<T> operator*(const Matrix3<T> &a, const Matrix3<T> &b) {
  return Matrix3<T>(
      a.n[0] * b.n[0] + a.n[1] * b.n[3] + a.n[2] * b.n[6], a.n[0] * b.n[1] + a.n[1] * b.n[4] + a.n[2] * b.n[7],
      a.n[0] * b.n[2] + a.n[1] * b.n[5] + a.n[2] * b.n[8], a.n[3] * b.n[0] + a.n[4] * b.n[3] + a.n[5] * b.n[6],
      a.n[3] * b.n[1] + a.n[4] * b.n[4] + a.n[5] * b.n[7], a.n[3] * b.n[2] + a.n[4] * b.n[5] + a.n[5] * b.n[8],
      a.n[6] * b.n[0] + a.n[7] * b.n[3] + a.n[8] * b.n[6], a.n[6] * b.n[1] + a.n[7] * b.n[4] + a.n[8] * b.n[7],
      a.n[6] * b.n[2] + a.n[7] * b.n[5] + a.n[8] * b.n[8]);
}

template <class T> inline Matrix3<T> operator*(const Matrix3<T> &a, const double d) {
  return Matrix3<T>(a.n[0] * d, a.n[1] * d, a.n[2] * d, a.n[3] * d, a.n[4] * d, a.n[5] * d, a.n[6] * d, a.n[7] * d,
                    a.n[8] * d);
}

template <class T> inline Matrix3<T> operator*(const double d, const Matrix3<T> &a) {
  return Matrix3<T>(a.n[0] * d, a.n[1] * d, a.n[2] * d, a.n[3] * d, a.n[4] * d, a.n[5] * d, a.n[6] * d, a.n[7] * d,
                    a.n[8] * d);
}

template <class T> inline Matrix3<T> operator/(const Matrix3<T> &a, const double d) {
  return Matrix3<T>(a.n[0] / d, a.n[1] / d, a.n[2] / d, a.n[3] / d, a.n[4] / d, a.n[5] / d, a.n[6] / d, a.n[7] / d,
                    a.n[8] / d);
}

template <class T>
inline AffineTransformMatrix<T> AffineTransformMatrix<T>::createRotation(T angle, float x, float y, float z) {
  double c = cos(angle);
  double s = sin(angle);
  double t = 1.0 - c;

  Vector3<T> a(x, y, z);
  a.normalize();
  return AffineTransformMatrix<T>(t * a[0] * a[0] + c, t * a[0] * a[1] - s * a[2], t * a[0] * a[2] + s * a[1], 0.0,
                                  t * a[0] * a[1] + s * a[2], t * a[1] * a[1] + c, t * a[1] * a[2] - s * a[0], 0.0,
                                  t * a[0] * a[2] - s * a[1], t * a[1] * a[2] + s * a[0], t * a[2] * a[2] + c, 0.0, 0.0,
                                  0.0, 0.0, 1.0);
}

template <class T> inline AffineTransformMatrix<T> AffineTransformMatrix<T>::createTranslation(T x, T y, T z) {
  AffineTransformMatrix<T> trans;

  trans[0][3] = x;
  trans[1][3] = y;
  trans[2][3] = z;

  return trans;
}

template <class T> inline AffineTransformMatrix<T> AffineTransformMatrix<T>::createScale(T sx, T sy, T sz) {
  AffineTransformMatrix<T> scale;
  scale[0][0] = sx;
  scale[1][1] = sy;
  scale[2][2] = sz;

  return scale;
}

template <class T> inline AffineTransformMatrix<T> AffineTransformMatrix<T>::createShear(T shx, T shy, T shz) {
  AffineTransformMatrix<T> shear;
  GVT_DEBUG(DBG_ALWAYS, "unimplemented matrix command createShear(shx,shy,shz)");
  return shear;
}

template <class T> inline Vector3<T> clamp(const Vector3<T> &other) {
  return maximum(Vector3<T>(), minimum(other, Vector3<T>(1.0, 1.0, 1.0)));
}

// These are handy functions

template <class T> inline void makeDiagonal(AffineTransformMatrix<T> &m, T k) {
  m[0][0] = k;
  m[0][1] = 0.0;
  m[0][2] = 0.0;
  m[0][3] = 0.0;
  m[1][0] = 0.0;
  m[1][1] = k;
  m[1][2] = 0.0;
  m[1][3] = 0.0;
  m[2][0] = 0.0;
  m[2][1] = 0.0;
  m[2][2] = k;
  m[2][3] = 0.0;
  m[3][0] = 0.0;
  m[3][1] = 0.0;
  m[3][2] = 0.0;
  m[3][3] = k;
}

template <class T> inline void makeHScale(AffineTransformMatrix<T> &m, T sx, T sy, T sz) {
  m[0][0] = sx;
  m[0][1] = 0.0;
  m[0][2] = 0.0;
  m[0][3] = 0.0;
  m[1][0] = 0.0;
  m[1][1] = sy;
  m[1][2] = 0.0;
  m[1][3] = 0.0;
  m[2][0] = 0.0;
  m[2][1] = 0.0;
  m[2][2] = sz;
  m[2][3] = 0.0;
  m[3][0] = 0.0;
  m[3][1] = 0.0;
  m[3][2] = 0.0;
  m[3][3] = 1.0;
}

template <class T> inline void makeHScale(AffineTransformMatrix<T> &m, const Vector3<T> &v) {
  makeHScale(m, v[0], v[1], v[2]);
}

template <class T> inline void makeHTrans(AffineTransformMatrix<T> &m, T tx, T ty, T tz) {
  m[0][0] = 1.0;
  m[0][1] = 0.0;
  m[0][2] = 0.0;
  m[0][3] = tx;
  m[1][0] = 0.0;
  m[1][1] = 1.0;
  m[1][2] = 0.0;
  m[1][3] = ty;
  m[2][0] = 0.0;
  m[2][1] = 0.0;
  m[2][2] = 1.0;
  m[2][3] = tz;
  m[3][0] = 0.0;
  m[3][1] = 0.0;
  m[3][2] = 0.0;
  m[3][3] = 1.0;
}

template <class T> inline void makeHTrans(AffineTransformMatrix<T> &m, const Vector3<T> &v) {
  makeHTrans(m, v[0], v[1], v[2]);
}

template <class T> inline void makeHRotX(AffineTransformMatrix<T> &m, T thetaX) {
  T cosT = (T)cos(thetaX);
  T sinT = (T)sin(thetaX);

  m[0][0] = 1.0;
  m[0][1] = 0.0;
  m[0][2] = 0.0;
  m[0][3] = 0.0;
  m[1][0] = 0.0;
  m[1][1] = cosT;
  m[1][2] = -sinT;
  m[1][3] = 0.0;
  m[2][0] = 0.0;
  m[2][1] = sinT;
  m[2][2] = cosT;
  m[2][3] = 0.0;
  m[3][0] = 0.0;
  m[3][1] = 0.0;
  m[3][2] = 0.0;
  m[3][3] = 1.0;
}

template <class T> inline void makeHRotY(AffineTransformMatrix<T> &m, T thetaY) {
  T cosT = (T)cos(thetaY);
  T sinT = (T)sin(thetaY);

  m[0][0] = cosT;
  m[0][1] = 0.0;
  m[0][2] = sinT;
  m[0][3] = 0.0;
  m[1][0] = 0.0;
  m[1][1] = 1.0;
  m[1][2] = 0.0;
  m[1][3] = 0.0;
  m[2][0] = -sinT;
  m[2][1] = 0.0;
  m[2][2] = cosT;
  m[2][3] = 0.0;
  m[3][0] = 0.0;
  m[3][1] = 0.0;
  m[3][2] = 0.0;
  m[3][3] = 1.0;
}

template <class T> inline void makeHRotZ(AffineTransformMatrix<T> &m, T thetaZ) {
  T cosT = (T)cos(thetaZ);
  T sinT = (T)sin(thetaZ);

  m[0][0] = cosT;
  m[0][1] = -sinT;
  m[0][2] = 0.0;
  m[0][3] = 0.0;
  m[1][0] = sinT;
  m[1][1] = cosT;
  m[1][2] = 0.0;
  m[1][3] = 0.0;
  m[2][0] = 0.0;
  m[2][1] = 0.0;
  m[2][2] = 1.0;
  m[2][3] = 0.0;
  m[3][0] = 0.0;
  m[3][1] = 0.0;
  m[3][2] = 0.0;
  m[3][3] = 1.0;
}
}
}
}

#include <gvt/core/math/MatrixOperation.inl>

#endif // GVT_CORE_MATH_MATRIX_H
