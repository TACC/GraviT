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
#ifndef GVT_RENDER_ADAPTER_OSPRAY_DATA_OSPAY_VOL_ADAPTER_H
#define GVT_RENDER_ADAPTER_OSPRAY_DATA_OSPAY_VOL_ADAPTER_H

#include "gvt/render/Adapter.h"
#include "gvt/render/data/primitives/Volume.h"
#include "gvt/render/adapter/ospray/data/OSPRayAdapter.h"


namespace gvt {
namespace render {
namespace adapter {
namespace ospray {
namespace data {
/** class to manage ospray rendering of a gravit volume mesh */
class OSPRayVolAdapter : public gvt::render::adapter::ospray::data::OSPRayAdapter {
public: // public stuff
  /**
   * Construct the Ospray mesh adapter. This constructor takes a volume argument. 
   * It also takes command line args. Ospray initialization requires these to parse
   * out the ospray related args. 
   */
  OSPRayVolAdapter(int *argc, char *argv[], gvt::render::data::primitives::Volume *vol);
  /**
   * Construct the Ospray adapter without passing in the command line args. The
   * constructor makes a dummy list and passes it to ospinit. 
   */
  OSPRayVolAdapter(gvt::render::data::primitives::Volume *vol);
  /**
   * Destruct the adapter
   */
  virtual ~OSPRayVolAdapter();
  /**
   * Trace using OSPRay
   */
  virtual void trace(gvt::render::actor::RayVector &rayList, gvt::render::actor::RayVector &moved_rays, glm::mat4 *m,
      glm::mat4 *minv, glm::mat3 *normi, std::vector<gvt::render::data::scene::Light *> &lights,
      size_t begin = 0, size_t end = 0);
  OSPVolume GetTheOSPVolume() {return theOSPVolume;}
protected: // protected stuff
  size_t begin, end;
  
  OSPVolume theOSPVolume;
  OSPData   theOSPData;
};
}
}
}
}
}

#endif //GVT_RENDER_ADAPTER_OSPRAY_DATA_OSPAY_VOL_ADAPTER_H
