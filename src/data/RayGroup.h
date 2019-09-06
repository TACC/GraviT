#ifndef GVT2_RAYGroup_H
#define GVT2_RAYGroup_H
/* ============================================================================
 * This file is released as part of GraviT2 - scalable, platform independent 
 * ray tracing tacc.github.io/GraviT2
 *
 * Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas
 * at Austin All rights reserved.
 *
 * Licensed under the BSD 3-Clause License, (the "License"); you may not use 
 * this file except in compliance with the License.
 * A copy of the License is included with this software in the file LICENSE.
 * If your copy does not contain the License, you may obtain a copy of the 
 * License at:
 *
 *     http://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and 
 * limitations under limitations under the License.
   ==========================================================================*/

#include "Ray.h"
#include <memory>
#include <float.h>
#include "vec.h"

namespace gvt2 {
/**
 * \brief A container class for a group of rays. 
 *
 * The group of rays is stored in a struct of arrays style to
 * accomodate vectorization. 
 */
class RayGroup {
    private:
        std::shared_ptr<double> ray_data;
        size_t size;
    public:
        ~RayGroup();
        RayGroup(size_t n);
};
}// end of namespace gvt2
#endif /* GVT2_RAYGroup_H */
