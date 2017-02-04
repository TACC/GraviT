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

#include <algorithm>

#include "DomainTracer.h"
#include "Messages/SendRayList.h"
#include <gvt/core/comm/communicator.h>

namespace gvt {
namespace render {

bool DomainTracer::areWeDone() {
  std::shared_ptr<gvt::comm::communicator> comm = gvt::comm::communicator::singleton();
  gvt::render::RenderContext &cntxt = *gvt::render::RenderContext::instance();
  std::shared_ptr<DomainTracer> tracer = std::dynamic_pointer_cast<DomainTracer>(cntxt.tracer());
  if (!tracer || tracer->getGlobalFrameFinished()) return false;
  bool ret = tracer->isDone();
  return ret;
}

void DomainTracer::Done(bool T) {
  std::shared_ptr<gvt::comm::communicator> comm = gvt::comm::communicator::singleton();
  gvt::render::RenderContext &cntxt = *gvt::render::RenderContext::instance();
  std::shared_ptr<DomainTracer> tracer = std::dynamic_pointer_cast<DomainTracer>(cntxt.tracer());
  if (!tracer) return;
  if (T) {
    // std::lock_guard<std::mutex> _lock(tracer->_queue._protect);
    // tracer->_queue._queue.clear();
    tracer->setGlobalFrameFinished(true);
  }
}
DomainTracer::DomainTracer() : gvt::render::RayTracer() {
  RegisterMessage<gvt::comm::EmptyMessage>();
  RegisterMessage<gvt::comm::SendRayList>();
  gvt::comm::communicator &comm = gvt::comm::communicator::instance();
  v = std::make_shared<comm::vote::vote>(DomainTracer::areWeDone, DomainTracer::Done);
  comm.setVote(v);

  queue_mutex = new std::mutex[meshRef.size()];
  for (auto &m : meshRef) {
    queue[m.first] = gvt::render::actor::RayVector();
  }

  instances_in_node.clear();
  gvt::core::Map<std::string, std::set<int> > remote_location;
  gvt::core::DBNodeH rootnode = cntxt->getRootNode();
  gvt::core::Vector<gvt::core::DBNodeH> dataNodes = rootnode["Data"].getChildren();
  // build location map, where meshes are by mpi node
  for (size_t i = 0; i < dataNodes.size(); i++) {

    gvt::core::Vector<gvt::core::DBNodeH> locations = dataNodes[i]["Locations"].getChildren();
    for (auto loc : locations) {
      remote_location[dataNodes[i].UUID().toString()].insert(loc.value().toInteger());
    }
  }

  gvt::core::Vector<gvt::core::DBNodeH> instancenodes = rootnode["Instances"].getChildren();
  for (int i = 0; i < instancenodes.size(); i++) {
    std::string UUID = instancenodes[i]["meshRef"].deRef().UUID().toString();
    if (remote_location[UUID].find(comm.id()) != remote_location[UUID].end()) {
      instances_in_node.insert(i);
    } else {
      remote[i] = remote_location[UUID];
    }
  }
}

DomainTracer::~DomainTracer() {
  if (queue_mutex != nullptr) delete[] queue_mutex;
  queue.clear();
}

void DomainTracer::resetBVH() {
  RayTracer::resetBVH();
  if (queue_mutex != nullptr) delete[] queue_mutex;
  for (auto &m : meshRef) {
    queue[m.first] = gvt::render::actor::RayVector();
  }
}

void DomainTracer::operator()() {
  gvt::comm::communicator &comm = gvt::comm::communicator::instance();
  _GlobalFrameFinished = false;
  img->reset();
  cam->AllocateCameraRays();
  cam->generateRays();
  processRaysAndDrop(cam->rays);
  gvt::render::actor::RayVector returned_rays;
  do {
    int target = -1;
    int amount = 0;
    for (auto &q : queue) {
      if (isInNode(q.first) && q.second.size() > amount) {
        amount = q.second.size();
        target = q.first;
      }
    }

    if (target != -1) {
      queue_mutex[target].lock();
      returned_rays.reserve(queue[target].size() * 10);
      calladapter(target, queue[target], returned_rays);
      queue[target].clear();
      queue_mutex[target].unlock();
      processRays(returned_rays, target);
    }

    if (target == -1) {
      for (auto q : queue) {
        if (isInNode(q.first) || q.second.empty()) continue;
        queue_mutex[q.first].lock();
        int sendto = pickNode(q.first);
        // if (sendto != comm.id() && sendto > 0) {
        std::shared_ptr<gvt::comm::Message> msg = std::make_shared<gvt::comm::SendRayList>(comm.id(), sendto, q.second);
        comm.send(msg, sendto);
        //}
        queue_mutex[q.first].unlock();
      }
    }

    if (isDone()) {
      v->PorposeVoting();
    }

  } while (hasWork());

  img->composite();
}

void DomainTracer::processRaysAndDrop(gvt::render::actor::RayVector &rays) {

  gvt::comm::communicator &comm = gvt::comm::communicator::instance();
  // const unsigned ray_chunk = rays.size() / comm.lastid();
  // const unsigned ray_start = ray_chunk * comm.id();
  // const unsigned ray_end = ray_chunk * (comm.id() + 1);

  const int chunksize =
      MAX(4096, rays.size() / (gvt::core::CoreContext::instance()->getRootNode()["threads"].value().toInteger() * 4));
  gvt::render::data::accel::BVH &acc = *bvh.get();
  static tbb::simple_partitioner ap;
  tbb::parallel_for(tbb::blocked_range<gvt::render::actor::RayVector::iterator>(rays.begin(), rays.end(), chunksize),
                    [&](tbb::blocked_range<gvt::render::actor::RayVector::iterator> raysit) {

                      gvt::core::Vector<gvt::render::data::accel::BVH::hit> hits =
                          acc.intersect<GVT_SIMD_WIDTH>(raysit.begin(), raysit.end(), -1);

                      gvt::core::Map<int, gvt::render::actor::RayVector> local_queue;
                      for (size_t i = 0; i < hits.size(); i++) {
                        gvt::render::actor::Ray &r = *(raysit.begin() + i);
                        if (hits[i].next != -1) {
                          r.origin = r.origin + r.direction * (hits[i].t * 0.95f);
                          if (isInNode(hits[i].next)) local_queue[hits[i].next].push_back(r);
                        }
                      }
                      for (auto &q : local_queue) {
                        queue_mutex[q.first].lock();
                        queue[q.first].insert(queue[q.first].end(),
                                              std::make_move_iterator(local_queue[q.first].begin()),
                                              std::make_move_iterator(local_queue[q.first].end()));
                        queue_mutex[q.first].unlock();
                      }
                    },
                    ap);

  rays.clear();
}

void DomainTracer::processRays(gvt::render::actor::RayVector &rays, const int src, const int dst) {

  const int chunksize =
      MAX(4096, rays.size() / (gvt::core::CoreContext::instance()->getRootNode()["threads"].value().toInteger() * 4));
  gvt::render::data::accel::BVH &acc = *bvh.get();
  static tbb::simple_partitioner ap;
  tbb::parallel_for(tbb::blocked_range<gvt::render::actor::RayVector::iterator>(rays.begin(), rays.end(), chunksize),
                    [&](tbb::blocked_range<gvt::render::actor::RayVector::iterator> raysit) {

                      gvt::core::Vector<gvt::render::data::accel::BVH::hit> hits =
                          acc.intersect<GVT_SIMD_WIDTH>(raysit.begin(), raysit.end(), src);

                      gvt::core::Map<int, gvt::render::actor::RayVector> local_queue;
                      for (size_t i = 0; i < hits.size(); i++) {
                        gvt::render::actor::Ray &r = *(raysit.begin() + i);
                        if (hits[i].next != -1) {
                          r.origin = r.origin + r.direction * (hits[i].t * 0.95f);
                          local_queue[hits[i].next].push_back(r);
                        } else if (r.type == gvt::render::actor::Ray::SHADOW && glm::length(r.color) > 0) {
                          img->localAdd(r.id, r.color * r.w, 1.f, r.t);
                        }
                      }
                      for (auto &q : local_queue) {
                        queue_mutex[q.first].lock();
                        queue[q.first].insert(queue[q.first].end(),
                                              std::make_move_iterator(local_queue[q.first].begin()),
                                              std::make_move_iterator(local_queue[q.first].end()));
                        queue_mutex[q.first].unlock();
                      }
                    },
                    ap);

  rays.clear();
}

bool DomainTracer::MessageManager(std::shared_ptr<gvt::comm::Message> msg) {
  std::shared_ptr<gvt::comm::communicator> comm = gvt::comm::communicator::singleton();
  gvt::render::actor::RayVector rays;
  rays.resize(msg->size() / sizeof(gvt::render::actor::Ray));
  std::memcpy(&rays[0], msg->getMessage<void>(), msg->size());
  processRays(rays);
  return true;
}

bool DomainTracer::isDone() {
  if (queue.empty()) return true;
  for (auto &q : queue)
    if (!q.second.empty()) return false;
  return true;
}
bool DomainTracer::hasWork() { return !_GlobalFrameFinished; }
}
}
