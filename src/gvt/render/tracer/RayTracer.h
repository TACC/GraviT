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

#ifndef GVT_RENDER_RAYTRACER
#define GVT_RENDER_RAYTRACER

#include <gvt/core/tracer/tracer.h>
#include <gvt/render/Adapter.h>
#include <gvt/render/RenderContext.h>
#include <gvt/render/Types.h>
#include <gvt/render/composite/IceTComposite.h>
#include <gvt/render/composite/ImageComposite.h>
#include <gvt/render/data/accel/BVH.h>
#include <gvt/render/data/scene/gvtCamera.h>

#include <tbb/blocked_range.h>
#include <tbb/mutex.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/partitioner.h>
#include <tbb/tick_count.h>

#ifdef GVT_RENDER_ADAPTER_EMBREE
#include <gvt/render/adapter/embree/EmbreeMeshAdapter.h>
#endif

#ifdef GVT_RENDER_ADAPTER_MANTA
#include <gvt/render/adapter/manta/MantaMeshAdapter.h>
#endif

#ifdef GVT_RENDER_ADAPTER_OPTIX
#include <gvt/render/adapter/optix/OptixMeshAdapter.h>
#endif

namespace gvt {
namespace render {

/**
 * \brief Ray tracer scheduler base class
 *
 * Implementation of base ray tracer scheduler to replace old scheduler implementation strategy. Previously the
 * scheduler was determined at compile time and could not be replaced at any time. The current implementation allows
 * the developer to replace the scheduler and queue management policy at any time.
 *
 * The current implementation also allows developers to more easily extend current schedulers instead of specializing
 * the template structure.
 *
 * It also uses the new strategy for Communication with each new scheduler implementing their own message types and
 * frame termination conditions through the voting mechanism.
 *
 */
class RayTracer : public gvt::core::Scheduler {
private:
protected:
  std::shared_ptr<gvt::render::composite::ImageComposite> img;  /**< Image compositing class **/
  std::shared_ptr<gvt::render::data::scene::gvtCameraBase> cam; /**< Camera class */
  std::shared_ptr<gvt::render::data::accel::BVH> bvh;           /**< Current global BVH structure */
  gvt::render::RenderContext *cntxt;                            /**< Current render context */

  // Scheduling
  std::mutex *queue_mutex = nullptr;                        /**< Multi thread queue protectiob */
  gvt::core::Map<int, gvt::render::actor::RayVector> queue; /**< Ray queue for each instance in the scene */

  // Caching
  gvt::core::Map<int, gvt::render::data::primitives::Mesh *> meshRef; /**< Map mesh internal id to pointer in memory */
  gvt::core::Map<int, glm::mat4 *> instM;                             /**< Mesh instance matrix model map */
  gvt::core::Map<int, glm::mat4 *> instMinv;                          /**< Mesh instance inverse matrix model map */
  gvt::core::Map<int, glm::mat3 *> instMinvN;                  /**< Mesh instance inverse matrix model map (3x3)*/
  gvt::core::Vector<gvt::render::data::scene::Light *> lights; /**< Scene lights */
  gvt::core::Map<gvt::render::data::primitives::Mesh *, std::shared_ptr<gvt::render::Adapter> >
      adapterCache /**< Tracer adapter cache */;
  int adapterType; /**< Current adapter type */

public:
  RayTracer();
  ~RayTracer();
  /**
   * \brief Scheduler implementation
   *
   * The scheduler implementation.
   *
   * @method operator
   */
  virtual void operator()();

  /**
   * \brief Call the adapter for a specific instance of a mesh
   *
   * Checks if the raytracer mesh adapter was already created and if so invokes the adapter trace function with the
   * correct arguments (Model, Inverse and Normal inverse matrices for the instance). If the adapter does not exist
   * invokes creates the adapter and places it in the cache.
   *
   * @method calladapter
   * @param  instTarget  Instance internal identifier
   * @param  toprocess   Rays to processed by the adapter
   * @param  moved_rays  Rays that escape the mesh AABB encoded in the adapter
   */
  void calladapter(const int instTarget, gvt::render::actor::RayVector &toprocess,
                   gvt::render::actor::RayVector &moved_rays);

  /**
   * Abstract method to process rays that where returned by the adapter call or a ray list list received from another
   * node
   * @method processRays
   * @param  rays        Rays returned by the adpter call
   * @param  src         Instance id from where the rays originated (-1 if camera rays)
   * @param  dst         Instance id to where the rays should be moved (dst >= 0 if sent from another node, note that
   * the rays where already processed)
   */
  virtual void processRays(gvt::render::actor::RayVector &rays, const int src = -1, const int dst = -1);

  /**
   * \brief Message Manager
   *
   * Then the communicatior receives a message it invokes the current scheduler to process it. This mmethod implements
   * the steps to process the message (e.g. if it is a ray list sent, it places the rays in the proper queue)
   *
   * @method MessageManager
   * @param  msg            Raw message received by the communicator
   * @return                True if the message was processed succefully
   */
  virtual bool MessageManager(std::shared_ptr<gvt::comm::Message> msg);

  /**
   * \brief Checks if the node scheduler is done
   *
   * Checks the termination condition according to the current scheduler implementation
   *
   * (e.g. if there is no more work in the queues and there are no messages to be processed or sent)
   * @method isDone
   * @return True if the node has/expects no more work
   */
  virtual bool isDone();
  /**
   * \brief Checks if the node scheduler has/expects work
   *
   * Checks the termination condition according to the current scheduler implementation
   *
   * (e.g. if there is no more work in the queues and there are no messages to be processed or sent)
   * @method isDone
   * @return True if the node has/expects no more work
   */
  virtual bool hasWork();
  /**
   * \brief Get the current image composite instance
   * @method getImageBuffer
   * @return [description]
   */
  virtual float *getImageBuffer();
  /**
   * \brief Resets camera
   *
   * Forces the camera object to read data from the render context and update itself
   *
   * @method resetCamera
   */
  virtual void resetCamera();
  /**
   * \brief Reset film
   *
   * Forces the image composite and the camera to update the film width and height from the render context
   *
   * @method resetFilm
   */
  virtual void resetFilm();
  /**
   * \brief Reset BVH
   *
   * Forces the BVH to check the render context for changes in the meshs and/or instances/
   *
   * @method resetBVH
   */
  virtual void resetBVH();
  /**
   * \brief Get current Image Composite instance
   * @return Current Image composite shared pointer
   */
  virtual std::shared_ptr<gvt::render::composite::ImageComposite> getComposite() { return img; }
};
};
};

#endif /*GVT_RENDER_RAYTRACER*/
