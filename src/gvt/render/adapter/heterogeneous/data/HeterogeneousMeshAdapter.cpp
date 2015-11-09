/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray
   tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas
   at Austin
   All rights reserved.

   Licensed under the BSD 3-Clause License, (the "License"); you may not use
   this file
   except in compliance with the License.
   A copy of the License is included with this software in the file LICENSE.
   If your copy does not contain the License, you may obtain a copy of the
   License at:

       http://opensource.org/licenses/BSD-3-Clause

   Unless required by applicable law or agreed to in writing, software
   distributed under
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY
   KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under
   limitations under the License.

   GraviT is funded in part by the US National Science Foundation under awards
   ACI-1339863,
   ACI-1339881 and ACI-1339840
   =======================================================================================
   */
#include "HeterogeneousMeshAdapter.h"
#include <thread>
#include <future>
#include <mutex>

using namespace gvt::render::actor;
using namespace gvt::render::adapter::heterogeneous::data;
using namespace gvt::render::data::primitives;
using namespace gvt::core::math;


HeterogeneousMeshAdapter::HeterogeneousMeshAdapter(gvt::core::DBNodeH node) : Adapter(node) {

   _embree = new gvt::render::adapter::embree::data::EmbreeMeshAdapter(node);
   _optix = new gvt::render::adapter::optix::data::OptixMeshAdapter(node);
 
}

HeterogeneousMeshAdapter::~HeterogeneousMeshAdapter() {
   delete _embree;
   delete _optix;
}

void HeterogeneousMeshAdapter::trace(gvt::render::actor::RayVector &rayList,
                             gvt::render::actor::RayVector &moved_rays,
                             gvt::core::DBNodeH instNode) {
#ifdef GVT_USE_DEBUG
 boost::timer::auto_cpu_timer t_functor("HeterogeneousMeshAdapter: trace time: %w\n");
#endif
  
  gvt::render::actor::RayVector rEmbree, rOptix;
  gvt::render::actor::RayVector mEmbree, mOptix;
  std::mutex _lock_moved_rays;

{
  std::copy(rayList.begin(), rayList.begin() + rayList.size() /2, std::back_inserter(rEmbree));
  std::copy(rayList.begin() + rayList.size() / 2, rayList.end(), std::back_inserter(rOptix));
}

{
   std::future<void> ef(std::async([&](){
      _embree->trace(rEmbree,mEmbree,instNode);
      std::lock_guard<std::mutex> lock(_lock_moved_rays);
      std::copy(mEmbree.begin(), mEmbree.end(), std::back_inserter(moved_rays));
   }));  
   std::future<void> of(std::async([&](){
      _optix->trace(rOptix,mOptix,instNode);
      std::lock_guard<std::mutex> lock(_lock_moved_rays);
      std::copy(mOptix.begin(), mOptix.end(), std::back_inserter(moved_rays));
   }));  
   ef.wait();
   of.wait();

}

  rayList.clear();
}
