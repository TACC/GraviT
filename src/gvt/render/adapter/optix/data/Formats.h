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
 * File:   optixdata.h
 * Author: jbarbosa
 *
 * Created on January 11, 2015, 10:24 PM
 */

#ifndef GVT_RENDER_ADAPTER_OPTIX_DATA_FORMATS_H
#define GVT_RENDER_ADAPTER_OPTIX_DATA_FORMATS_H

namespace gvt {
namespace render {
namespace adapter {
namespace optix {
namespace data {

/// OptiX ray format
struct OptixRay {
  float origin[3];
  float t_min;
  float direction[3];
  float t_max;
  friend std::ostream &operator<<(std::ostream &os, const OptixRay &r) {
    return (os << "ray  o: " << r.origin[0] << ", " << r.origin[1] << ", " << r.origin[2] << " d: " << r.direction[0]
               << ", " << r.direction[1] << ", " << r.direction[2]);
  }
};

/// OptiX hit format
struct OptixHit {
  float t;
  int triangle_id;
  float u;
  float v;
  friend std::ostream &operator<<(std::ostream &os, const OptixHit &oh) {
    return (os << "hit  t: " << oh.t << " triID: " << oh.triangle_id);
  }
};
}
}
}
}
}

#endif /* GVT_RENDER_ADAPTER_OPTIX_DATA_FORMATS_H */
