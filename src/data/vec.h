/* ========================================================================== //
 * This file is released as part of GraviT2 - scalable, platform independent  //
 * ray tracing tacc.github.io/GraviT2                                         //
 *                                                                            //
 * Copyright (c) 2013-2019 The University of Texas at Austin.                 //
 * All rights reserved.                                                       //
 *                                                                            //
 * Licensed under the Apache License, Version 2.0 (the "License");            //
 * you may not use this file except in compliance with the License.           //
 * A copy of the License is included with this software in the file LICENSE.  //
 * If your copy does not contain the License, you may obtain a copy of the    //
 * License at:                                                                //
 *                                                                            //
 *     https://www.apache.org/licenses/LICENSE-2.0                            //
 *                                                                            //
 * Unless required by applicable law or agreed to in writing, software        //
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT  //
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.           //
 * See the License for the specific language governing permissions and        //
 * limitations under the License.                                             //
 * ========================================================================== */

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
            v[0] = v1; v[1] = v2; v[2] = v3;
        };
        vec3f() { v[0] = 0.;v[1]=0.;v[2]=0.; };
        vec3f(const vec4f vin) {v[0]=vin[0];v[1]=vin[1];v[2]=vin[2];};
        inline float operator[](int i) const {return v[i];};
        inline float& operator[](int i) {return v[i];};
};
class vec4f {
    private:
        float v[4];
    public:
        vec4f(const float v1,const float v2, const float v3, const float v4) {
            v[0]=v1; v[1]=v2; v[2]=v3; v[3]=v4;
        };
        vec4f() {v[0]=0.;v[1]=0.;v[2]=0.;v[3]=0.;};
        inline float operator[](int i) const {return v[i];};
        inline float& operator[](int i) {return v[i];};
};
class mat4f {
    private:
        float m[16];
    public:
        mat4f() {for(int i=0;i<16;i++) m[i]=0.0;}
        identity() {mat4f();for(int i=0;i<4;i++) m[i*5]=1.0;}
};
}
#endif /* GVT2_VEC_H */
