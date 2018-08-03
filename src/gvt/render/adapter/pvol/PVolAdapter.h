/* ======================================================================================= 
   This file is released as part of GraviT - scalable, platform independent ray tracing
   tacc.github.io/GraviT

   Copyright 2013-2018 Texas Advanced Computing Center, The University of Texas at Austin  
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
#ifndef GVT_RENDER_ADAPTER_PVOL_ADAPTER_H
#define GVT_RENDER_ADAPTER_PVOL_ADAPTER_H

#include <iostream>
#include "gvt/render/Adapter.h"
#include <gvt/render/cntx/rcontext.h>
#include <pvol.h>
#include <dtypes.h>
#include <Lighting.h>
#include <Rays.h>
#include <Renderer.h>
#include <Visualization.h>
#include <Volume.h>
#include <VolumeVis.h>

namespace gvt {
namespace render {
namespace adapter {
namespace pvol {
namespace data {

class PVolAdapter : public gvt::render::Adapter {
public:
  /** 
   * Construct the PVolAdapter. 
   */
  PVolAdapter(std::shared_ptr<gvt::render::data::primitives::Data> data,int w, int h);
  ::pvol::Application* GetTheApplication() {return theApplication;}
  ::pvol::VisualizationP GetTheVisualization() {return theVisualization;}
  ::pvol::Lighting GetTheLighting() {return theLighting;}
  ~PVolAdapter();
  static void init_pvol(int *argc,char **argv) ;
  ::pvol::RayList GVT2PVOLRays(gvt::render::actor::RayVector &rayList);
  void PVOL2GVTMoved_Rays(::pvol::RayList *out, ::pvol::RayList& rl, gvt::render::actor::RayVector& moved_rays);
  virtual void trace( gvt::render::actor::RayVector &rayList, 
                      gvt::render::actor::RayVector &moved_rays, 
                      glm::mat4 *m,
                      glm::mat4 *minv, 
                      glm::mat3 *normi, 
                      gvt::core::Vector<std::shared_ptr<gvt::render::data::scene::Light> > &lights,
                      size_t begin = 0, 
                      size_t end = 0);

protected:
  /**
   * Variable: adapter has been initialized if true. 
   */
  static bool init;
  size_t begin, end;
  int width, height;
  bool dolights;
public:
  static ::pvol::Application*   theApplication;
  static ::pvol::RendererP theRenderer;
  ::pvol::DatasetsP theDataset;
  ::pvol::Lighting       theLighting;
  ::pvol::VisualizationP theVisualization;
};
}
}
}
}
}
#endif // GVT_RENDER_ADAPTER_PVOL_ADAPTER_H
