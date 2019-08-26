
#ifndef GVT2_VEC_H
#define GVT2_VEC_H

/**
 * \brief A local set of mathematical vectors
 */

namespace gvt2 {
class vec3f {
    private:
        float v[3];
    public:
        vec3f(const float v1,const float v2, const float v3) {
            v[0] = v1; v[1] = v2; v[3] = v3;
        };
        vec3f() { v[0] = 0.;v[1]=0.;v[2]=0.; };
        inline float operator[](int i) const {return v[i];};
        inline float& operator[](int i) {return v[i];};
};
}
#endif /* GVT2_VEC_H */
