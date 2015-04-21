/* 
 * File:   MatrixOperation.inl
 * Author: jbarbosa
 *
 * Created on May 3, 2014, 1:05 PM
 */

template <class T>
inline bool operator ==(const GVT::Math::Matrix3<T>& a, const GVT::Math::Matrix3<T>& b) {
    return !memcmp(a.n, b.n, 9 * sizeof (T));
}

template <class T>
inline bool operator !=(const GVT::Math::Matrix3<T>& a, const GVT::Math::Matrix3<T>& b) {
    return memcmp(a.n, b.n, 9 * sizeof (T));
}

template <class T>
inline std::ostream& operator <<(std::ostream& os, const GVT::Math::Matrix3<T>& m) {
    os << m.n[0] << " " << m.n[1] << " " << m.n[2];
    os << m.n[3] << " " << m.n[4] << " " << m.n[5];
    os << m.n[6] << " " << m.n[7] << " " << m.n[8];
}

template <class T>
inline std::istream& operator >>(std::istream& is, GVT::Math::Matrix3<T>& m) {
    is >> m.n[0] >> m.n[1] >> m.n[2];
    is >> m.n[3] >> m.n[4] >> m.n[5];
    is >> m.n[6] >> m.n[7] >> m.n[8];
}

template <class T>
inline GVT::Math::AffineTransformMatrix<T> operator -(const GVT::Math::AffineTransformMatrix<T>& a) {
    return GVT::Math::AffineTransformMatrix<T>(-a.n[ 0], -a.n[ 1], -a.n[ 2], -a.n[ 3],
            -a.n[ 4], -a.n[ 5], -a.n[ 6], -a.n[ 7],
            -a.n[ 8], -a.n[ 9], -a.n[10], -a.n[11],
            -a.n[12], -a.n[13], -a.n[14], -a.n[15]);
}

template <class T>
inline GVT::Math::AffineTransformMatrix<T> operator +(const GVT::Math::AffineTransformMatrix<T>& a, const GVT::Math::AffineTransformMatrix<T>& b) {
    return GVT::Math::AffineTransformMatrix<T>(a.n[ 0] + b.n[ 0], a.n[ 1] + b.n[ 1], a.n[ 2] + b.n[ 2], a.n[ 3] + b.n[ 3],
            a.n[ 4] + b.n[ 4], a.n[ 5] + b.n[ 5], a.n[ 6] + b.n[ 6], a.n[ 7] + b.n[ 7],
            a.n[ 8] + b.n[ 8], a.n[ 9] + b.n[ 9], a.n[10] + b.n[10], a.n[11] + b.n[11],
            a.n[12] + b.n[12], a.n[13] + b.n[13], a.n[14] + b.n[14], a.n[15] + b.n[15]);
}

template <class T>
inline GVT::Math::AffineTransformMatrix<T> operator -(const GVT::Math::AffineTransformMatrix<T>& a, const GVT::Math::AffineTransformMatrix<T>& b) {
    return GVT::Math::AffineTransformMatrix<T>(a.n[ 0] - b.n[ 0], a.n[ 1] - b.n[ 1], a.n[ 2] - b.n[ 2], a.n[ 3] - b.n[ 3],
            a.n[ 4] - b.n[ 4], a.n[ 5] - b.n[ 5], a.n[ 6] - b.n[ 6], a.n[ 7] - b.n[ 7],
            a.n[ 8] - b.n[ 8], a.n[ 9] - b.n[ 9], a.n[10] - b.n[10], a.n[11] - b.n[11],
            a.n[12] - b.n[12], a.n[13] - b.n[13], a.n[14] - b.n[14], a.n[15] - b.n[15]);
}

template <class T>
inline GVT::Math::AffineTransformMatrix<T> operator *(const GVT::Math::AffineTransformMatrix<T>& a, const GVT::Math::AffineTransformMatrix<T>& b) {
    return GVT::Math::AffineTransformMatrix<T>(a.n[ 0] * b.n[ 0] + a.n[ 1] * b.n[ 4] + a.n[ 2] * b.n[ 8] + a.n[ 3] * b.n[12],
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
inline GVT::Math::AffineTransformMatrix<T> operator *(const GVT::Math::AffineTransformMatrix<T>& a, const double d) {
    return GVT::Math::AffineTransformMatrix<T>(a.n[ 0] * d, a.n[ 1] * d, a.n[ 2] * d, a.n[ 3] * d,
            a.n[ 4] * d, a.n[ 5] * d, a.n[ 6] * d, a.n[ 7] * d,
            a.n[ 8] * d, a.n[ 9] * d, a.n[10] * d, a.n[11] * d,
            a.n[12] * d, a.n[13] * d, a.n[14] * d, a.n[15] * d);
}

template <class T>
inline GVT::Math::AffineTransformMatrix<T> operator *(const double d, const GVT::Math::AffineTransformMatrix<T>& a) {
    return GVT::Math::AffineTransformMatrix<T>(a.n[ 0] * d, a.n[ 1] * d, a.n[ 2] * d, a.n[ 3] * d,
            a.n[ 4] * d, a.n[ 5] * d, a.n[ 6] * d, a.n[ 7] * d,
            a.n[ 8] * d, a.n[ 9] * d, a.n[10] * d, a.n[11] * d,
            a.n[12] * d, a.n[13] * d, a.n[14] * d, a.n[15] * d);
}

template <class T>
inline GVT::Math::AffineTransformMatrix<T> operator /(const GVT::Math::AffineTransformMatrix<T>& a, const double d) {
    return GVT::Math::AffineTransformMatrix<T>(a.n[ 0] / d, a.n[ 1] / d, a.n[ 2] / d, a.n[ 3] / d,
            a.n[ 4] / d, a.n[ 5] / d, a.n[ 6] / d, a.n[ 7] / d,
            a.n[ 8] / d, a.n[ 9] / d, a.n[10] / d, a.n[11] / d,
            a.n[12] / d, a.n[13] / d, a.n[14] / d, a.n[15] / d);
}

template <class T>
inline bool operator ==(const GVT::Math::AffineTransformMatrix<T>& a, const GVT::Math::AffineTransformMatrix<T>& b) {
    return !memcmp(a.n, b.n, 16 * sizeof (T));
}

template <class T>
inline bool operator !=(const GVT::Math::AffineTransformMatrix<T>& a, const GVT::Math::AffineTransformMatrix<T>& b) {
    return memcmp(a.n, b.n, 16 * sizeof (T));
}

template <class T>
inline std::ostream& operator <<(std::ostream& os, const GVT::Math::AffineTransformMatrix<T>& m) {
    return os << m.n[ 0] << " " << m.n[ 1] << " " << m.n[ 2] << " " << m.n[ 3] << std::endl <<
            m.n[ 4] << " " << m.n[ 5] << " " << m.n[ 6] << " " << m.n[ 7] << std::endl <<
            m.n[ 8] << " " << m.n[ 9] << " " << m.n[10] << " " << m.n[11] << std::endl <<
            m.n[12] << " " << m.n[13] << " " << m.n[14] << " " << m.n[15] << std::endl;
}

template <class T>
inline std::istream& operator >>(std::istream& is, GVT::Math::AffineTransformMatrix<T>& m) {
    is >> m.n[ 0] >> m.n[ 1] >> m.n[ 2] >> m.n[ 3];
    is >> m.n[ 4] >> m.n[ 5] >> m.n[ 6] >> m.n[ 7];
    is >> m.n[ 8] >> m.n[ 9] >> m.n[10] >> m.n[11];
    is >> m.n[12] >> m.n[13] >> m.n[14] >> m.n[15];
}

