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
#include <cmath>

using namespace gvt::render::actor;
using namespace gvt::render::adapter::heterogeneous::data;
using namespace gvt::render::data::primitives;
using namespace gvt::core::math;

HeterogeneousMeshAdapter::HeterogeneousMeshAdapter(gvt::core::DBNodeH node)
    : Adapter(node) {
  _embree = new gvt::render::adapter::embree::data::EmbreeMeshAdapter(node);
  _optix = new gvt::render::adapter::optix::data::OptixMeshAdapter(node);
}

HeterogeneousMeshAdapter::~HeterogeneousMeshAdapter() {
  delete _embree;
  delete _optix;
}

void HeterogeneousMeshAdapter::trace(gvt::render::actor::RayVector &rayList,
                                     gvt::render::actor::RayVector &moved_rays,
                                     gvt::core::DBNodeH instNode, size_t begin,
                                     size_t end) {
#ifdef GVT_USE_DEBUG
  boost::timer::auto_cpu_timer t_functor(
      "HeterogeneousMeshAdapter: trace time: %w\n");
#endif

  gvt::render::actor::RayVector rEmbree, rOptix;
  gvt::render::actor::RayVector mEmbree, mOptix;
  std::mutex _lock_rays;

  {
    const size_t size = rayList.size();
    const size_t work = std::min( (size_t)(16*1024), size / 2);
    size_t current = 0;

    std::atomic<size_t> cput(0), gput(0);

    std::future<void> ef(std::async([&]() {
      while (current < size) {
        if (_lock_rays.try_lock()) {
          size_t start = current;
          current += work;
          size_t end = current;
          _lock_rays.unlock();

          if(start >= size) continue;
          if (end >= size)
            end = size;

          _embree->trace(rayList, moved_rays, instNode, start, end);
          cput++;
        }
      }
    }));

    std::future<void> of(std::async([&]() {
      while (current < size) {
        if (_lock_rays.try_lock()) {
          size_t start = current;
          current += work;
          size_t end = current;
          _lock_rays.unlock();

          if(start >= size) continue;
          if (end >= size)
            end = size;

          _optix->trace(rayList, mOptix, instNode, start, end);
          gput++;
        }
      }

    }));
    ef.wait();
    of.wait();
    moved_rays.insert(moved_rays.end(), std::make_move_iterator(mOptix.begin()),
                      std::make_move_iterator(mOptix.end()));

    std::cout << "C: " << cput << " G: " << gput << std::endl;
  }

  // rayList.clear();
}
