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
#ifndef GVT2_RAY_H
#define GVT2_RAY_H

#include <float.h>
#include "vec.h"

namespace gvt2 {
/**
 * \brief A single Ray class. 
 */
class Ray {
    public:
            vec3f origin;
            vec3f direction;
            float t_min; 
            float t_max;
            float t;
            int id;
            int type;
    public:
        Ray(const vec3f& origin, const vec3f& direction) ;
};
}// end of namespace gvt2
#endif /* GVT2_RAY_H */
