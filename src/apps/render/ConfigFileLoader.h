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
 * Author: jbarbosa
 *
 * Created on January 21, 2015, 12:15 PM
 */

#ifndef GVTAPPS_RENDER_CONFIG_FILE_LOADER_H
#define GVTAPPS_RENDER_CONFIG_FILE_LOADER_H

#include <string>

#include <gvt/core/Debug.h>
#include <gvt/render/data/Domains.h>

namespace gvtapps {
namespace render {

/// GVT configuration file loader
/** Load scene data from a gvt configuration file. The configuration file
 contains "scene" information such as camera descriptions, lights,
 and descriptions of geometric objects in the scene. The components are loaded
 into a render dataset object. The config file that is read
 by this class also contains information on which renderer to use. Back end
 renderers such as Optix and Manta do the actual ray-geometry
 intersections. The configuration file loader also adapts to object types and
 uses the appropriate reader class to parse the geometric
 input. The extension of the file name in the config file is used to select
 which geometry reader to use. For example an .obj file extension
 would cause the gvt obj file reader to be used to parse that file.
 File:   ConfigFileLoader.h
*/
class ConfigFileLoader {
public:
  /** Constructor that utilizes the file name of the gvt config file.
  */
  ConfigFileLoader(const std::string filename = "");
  /** Copy constructor.
  */
  ConfigFileLoader(const ConfigFileLoader &orig);
  virtual ~ConfigFileLoader();

  // clang-format off
  enum AccelType {
    NoAccel,
    BVH
  };
  // clang-format on

  /** gvt render dataset member function that contains all the scene data,
  cameras, lights, objects etc.
  */
  // gvt::render::data::Dataset scene;

  /**  data member that indicates the type of renderer to use. Options include
  but are not limited to Manta, Optix, and Embree.
  */
  int domain_type = 0; /* default Manta domain */
  /**  data member that indicates the type of scheduler to use. The default
  scheduler is the Image scheduler
  */
  int scheduler_type = 0; /* default Image scheduler */
  /**  data member that indicates the type of acceleration structure to use. The
  default scheduler is without acceleration.
  0: NONE, 1: BVH
  */
  AccelType accel_type = NoAccel; /* default no acceleration */
};
}
}

#endif /* GVTAPPS_RENDER_CONFIG_FILE_LOADER_H */
