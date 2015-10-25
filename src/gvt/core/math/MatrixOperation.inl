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
 * File:   MatrixOperation.inl
 * Author: jbarbosa
 *
 * Created on May 3, 2014, 1:05 PM
 */

template <class T>
inline bool operator ==(const gvt::core::math::Matrix3<T>& a, const gvt::core::math::Matrix3<T>& b) {
    return !memcmp(a.n, b.n, 9 * sizeof (T));
}

template <class T>
inline bool operator !=(const gvt::core::math::Matrix3<T>& a, const gvt::core::math::Matrix3<T>& b) {
    return memcmp(a.n, b.n, 9 * sizeof (T));
}

template <class T>
inline std::ostream& operator <<(std::ostream& os, const gvt::core::math::Matrix3<T>& m) {
    os << m.n[0] << " " << m.n[1] << " " << m.n[2];
    os << m.n[3] << " " << m.n[4] << " " << m.n[5];
    os << m.n[6] << " " << m.n[7] << " " << m.n[8];
    return os;
}

template <class T>
inline std::istream& operator >>(std::istream& is, gvt::core::math::Matrix3<T>& m) {
    is >> m.n[0] >> m.n[1] >> m.n[2];
    is >> m.n[3] >> m.n[4] >> m.n[5];
    is >> m.n[6] >> m.n[7] >> m.n[8];
    return is;
}

template <class T>
inline gvt::core::math::AffineTransformMatrix<T> operator -(const gvt::core::math::AffineTransformMatrix<T>& a) {
    return gvt::core::math::AffineTransformMatrix<T>(-a.n[ 0], -a.n[ 1], -a.n[ 2], -a.n[ 3],
            -a.n[ 4], -a.n[ 5], -a.n[ 6], -a.n[ 7],
            -a.n[ 8], -a.n[ 9], -a.n[10], -a.n[11],
            -a.n[12], -a.n[13], -a.n[14], -a.n[15]);
}

template <class T>
inline gvt::core::math::AffineTransformMatrix<T> operator +(const gvt::core::math::AffineTransformMatrix<T>& a, const gvt::core::math::AffineTransformMatrix<T>& b) {
    return gvt::core::math::AffineTransformMatrix<T>(a.n[ 0] + b.n[ 0], a.n[ 1] + b.n[ 1], a.n[ 2] + b.n[ 2], a.n[ 3] + b.n[ 3],
            a.n[ 4] + b.n[ 4], a.n[ 5] + b.n[ 5], a.n[ 6] + b.n[ 6], a.n[ 7] + b.n[ 7],
            a.n[ 8] + b.n[ 8], a.n[ 9] + b.n[ 9], a.n[10] + b.n[10], a.n[11] + b.n[11],
            a.n[12] + b.n[12], a.n[13] + b.n[13], a.n[14] + b.n[14], a.n[15] + b.n[15]);
}

template <class T>
inline gvt::core::math::AffineTransformMatrix<T> operator -(const gvt::core::math::AffineTransformMatrix<T>& a, const gvt::core::math::AffineTransformMatrix<T>& b) {
    return gvt::core::math::AffineTransformMatrix<T>(a.n[ 0] - b.n[ 0], a.n[ 1] - b.n[ 1], a.n[ 2] - b.n[ 2], a.n[ 3] - b.n[ 3],
            a.n[ 4] - b.n[ 4], a.n[ 5] - b.n[ 5], a.n[ 6] - b.n[ 6], a.n[ 7] - b.n[ 7],
            a.n[ 8] - b.n[ 8], a.n[ 9] - b.n[ 9], a.n[10] - b.n[10], a.n[11] - b.n[11],
            a.n[12] - b.n[12], a.n[13] - b.n[13], a.n[14] - b.n[14], a.n[15] - b.n[15]);
}

template <class T>
inline gvt::core::math::AffineTransformMatrix<T> operator *(const gvt::core::math::AffineTransformMatrix<T>& a, const gvt::core::math::AffineTransformMatrix<T>& b) {
    return gvt::core::math::AffineTransformMatrix<T>(a.n[ 0] * b.n[ 0] + a.n[ 1] * b.n[ 4] + a.n[ 2] * b.n[ 8] + a.n[ 3] * b.n[12],
            a.n[ 0] * b.n[ 1] + a.n[ 1] * b.n[ 5] + a.n[ 2] * b.n[ 9] + a.n[ 3] * b.n[13],
            a.n[ 0] * b.n[ 2] + a.n[ 1] * b.n[ 6] + a.n[ 2] * b.n[10] + a.n[ 3] * b.n[14],
            a.n[ 0] * b.n[ 3] + a.n[ 1] * b.n[ 7] + a.n[ 2] * b.n[11] + a.n[ 3] * b.n[15],
            a.n[ 4] * b.n[ 0] + a.n[ 5] * b.n[ 4] + a.n[ 6] * b.n[ 8] + a.n[ 7] * b.n[12],
            a.n[ 4] * b.n[ 1] + a.n[ 5] * b.n[ 5] + a.n[ 6] * b.n[ 9] + a.n[ 7] * b.n[13],
            a.n[ 4] * b.n[ 2] + a.n[ 5] * b.n[ 6] + a.n[ 6] * b.n[10] + a.n[ 7] * b.n[14],
            a.n[ 4] * b.n[ 3] + a.n[ 5] * b.n[ 7] + a.n[ 6] * b.n[11] + a.n[ 7] * b.n[15],
            a.n[ 8] * b.n[ 0] + a.n[ 9] * b.n[ 4] + a.n[10] * b.n[ 8] + a.n[11] * b.n[12],
            a.n[ 8] * b.n[ 1] + a.n[ 9] * b.n[ 5] + a.n[10] * b.n[ 9] + a.n[11] * b.n[13],
            a.n[ 8] * b.n[ 2] + a.n[ 9] * b.n[ 6] + a.n[10] * b.n[10] + a.n[11] * b.n[14],
            a.n[ 8] * b.n[ 3] + a.n[ 9] * b.n[ 7] + a.n[10] * b.n[11] + a.n[11] * b.n[15],
            a.n[12] * b.n[ 0] + a.n[13] * b.n[ 4] + a.n[14] * b.n[ 8] + a.n[15] * b.n[12],
            a.n[12] * b.n[ 1] + a.n[13] * b.n[ 5] + a.n[14] * b.n[ 9] + a.n[15] * b.n[13],
            a.n[12] * b.n[ 2] + a.n[13] * b.n[ 6] + a.n[14] * b.n[10] + a.n[15] * b.n[14],
            a.n[12] * b.n[ 3] + a.n[13] * b.n[ 7] + a.n[14] * b.n[11] + a.n[15] * b.n[15]);
}

template <class T>
inline gvt::core::math::AffineTransformMatrix<T> operator *(const gvt::core::math::AffineTransformMatrix<T>& a, const double d) {
    return gvt::core::math::AffineTransformMatrix<T>(a.n[ 0] * d, a.n[ 1] * d, a.n[ 2] * d, a.n[ 3] * d,
            a.n[ 4] * d, a.n[ 5] * d, a.n[ 6] * d, a.n[ 7] * d,
            a.n[ 8] * d, a.n[ 9] * d, a.n[10] * d, a.n[11] * d,
            a.n[12] * d, a.n[13] * d, a.n[14] * d, a.n[15] * d);
}

template <class T>
inline gvt::core::math::AffineTransformMatrix<T> operator *(const double d, const gvt::core::math::AffineTransformMatrix<T>& a) {
    return gvt::core::math::AffineTransformMatrix<T>(a.n[ 0] * d, a.n[ 1] * d, a.n[ 2] * d, a.n[ 3] * d,
            a.n[ 4] * d, a.n[ 5] * d, a.n[ 6] * d, a.n[ 7] * d,
            a.n[ 8] * d, a.n[ 9] * d, a.n[10] * d, a.n[11] * d,
            a.n[12] * d, a.n[13] * d, a.n[14] * d, a.n[15] * d);
}

template <class T>
inline gvt::core::math::AffineTransformMatrix<T> operator /(const gvt::core::math::AffineTransformMatrix<T>& a, const double d) {
    return gvt::core::math::AffineTransformMatrix<T>(a.n[ 0] / d, a.n[ 1] / d, a.n[ 2] / d, a.n[ 3] / d,
            a.n[ 4] / d, a.n[ 5] / d, a.n[ 6] / d, a.n[ 7] / d,
            a.n[ 8] / d, a.n[ 9] / d, a.n[10] / d, a.n[11] / d,
            a.n[12] / d, a.n[13] / d, a.n[14] / d, a.n[15] / d);
}

template <class T>
inline bool operator ==(const gvt::core::math::AffineTransformMatrix<T>& a, const gvt::core::math::AffineTransformMatrix<T>& b) {
    return !memcmp(a.n, b.n, 16 * sizeof (T));
}

template <class T>
inline bool operator !=(const gvt::core::math::AffineTransformMatrix<T>& a, const gvt::core::math::AffineTransformMatrix<T>& b) {
    return memcmp(a.n, b.n, 16 * sizeof (T));
}

template <class T>
inline std::ostream& operator <<(std::ostream& os, const gvt::core::math::AffineTransformMatrix<T>& m) {
    return os << m.n[ 0] << " " << m.n[ 1] << " " << m.n[ 2] << " " << m.n[ 3] << std::endl <<
            m.n[ 4] << " " << m.n[ 5] << " " << m.n[ 6] << " " << m.n[ 7] << std::endl <<
            m.n[ 8] << " " << m.n[ 9] << " " << m.n[10] << " " << m.n[11] << std::endl <<
            m.n[12] << " " << m.n[13] << " " << m.n[14] << " " << m.n[15] << std::endl;
}

template <class T>
inline std::istream& operator >>(std::istream& is, gvt::core::math::AffineTransformMatrix<T>& m) {
    is >> m.n[ 0] >> m.n[ 1] >> m.n[ 2] >> m.n[ 3];
    is >> m.n[ 4] >> m.n[ 5] >> m.n[ 6] >> m.n[ 7];
    is >> m.n[ 8] >> m.n[ 9] >> m.n[10] >> m.n[11];
    is >> m.n[12] >> m.n[13] >> m.n[14] >> m.n[15];
    return is;
}

