/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray
   tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas at Austin
   All rights reserved.

   Licensed under the BSD 3-Clause License, (the "License"); you may not use this file
   except in compliance with the License.
   A copy of the License is included with this software in the file LICENSE.
   If your copy does not contain the License, you may obtain a copy of the
   License at:

       http://opensource.org/licenses/BSD-3-Clause

   Unless required by applicable law or agreed to in writing, software distributed under
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.
   See the License for the specific language governing permissions and limitations under
   limitations under the License.

   GraviT is funded in part by the US National Science Foundation under awards
   ACI-1339863,
   ACI-1339881 and ACI-1339840
   =======================================================================================
   */
#include "HeterogeneousMeshAdapter.h"
#include <atomic>
#include <cmath>
#include <future>
#include <mutex>
#include <thread>

#include <tbb/task_group.h>

using namespace gvt::render::actor;
using namespace gvt::render::adapter::heterogeneous::data;
//using namespace gvt::render::data::primitives;

HeterogeneousMeshAdapter::HeterogeneousMeshAdapter(gvt::render::data::primitives::Mesh *mesh) : Adapter(mesh) {
  _embree = new gvt::render::adapter::embree::data::EmbreeMeshAdapter(mesh);
  _optix = new gvt::render::adapter::optix::data::OptixMeshAdapter(mesh);
}

HeterogeneousMeshAdapter::~HeterogeneousMeshAdapter() {
  delete _embree;
  delete _optix;
}

void HeterogeneousMeshAdapter::trace(gvt::render::actor::RayVector &rayList,
		gvt::render::actor::RayVector &moved_rays, glm::mat4 *m,
		glm::mat4 *minv, glm::mat3 *normi,
		std::vector<gvt::render::data::scene::Light *> &lights, size_t begin,
		size_t end) {

	gvt::render::actor::RayVector mOptix;
	std::mutex _lock_rays;

	const size_t size = rayList.size();
	bool useGPU = true;

	if (size < _optix->packetSize / 2) /* hand tunned value*/
		useGPU = false;

	size_t split = end;
	if (useGPU)
		split = size * 0.60;


	tbb::task_group g;

	g.run([&]() {
		_embree->trace(rayList, moved_rays, m, minv, normi, lights, 0, split);

	});

	if (useGPU)
		g.run([&]() {
			_optix->trace(rayList, mOptix, m, minv, normi, lights, split, end);

		});

	g.wait();
	if (useGPU) {
		moved_rays.reserve(moved_rays.size() + mOptix.size());
		moved_rays.insert(moved_rays.end(),
				std::make_move_iterator(mOptix.begin()),
				std::make_move_iterator(mOptix.end()));
	}

}
