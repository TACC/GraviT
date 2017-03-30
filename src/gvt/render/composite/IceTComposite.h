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

   GraviT is funded in part by the US National Science Foundation under awards
   ACI-1339863,
   ACI-1339881 and ACI-1339840
   =======================================================================================
   */

#ifndef GVT_ICET_COMPOSITE_H
#define GVT_ICET_COMPOSITE_H

#include <IceT.h>
#include <gvt/core/composite/Composite.h>
#include <gvt/render/composite/ImageComposite.h>
namespace gvt {
namespace render {
namespace composite {

/**
 * @brief buffer composite using IceT
 *
 * Uses IceT (http://icet.sandia.gov) framework to compose the final image
 *
 */
struct IceTComposite : gvt::render::composite::ImageComposite {

  // glm::vec4 *buffer;
  IceTInt num_proc;              /**< IceT number of mpi processes */
  IceTFloat *color_buffer;       /**< IceT compute node local color buffer */
  IceTFloat *color_buffer_final; /**< IceT final (composited) buffer */
  IceTFloat *depth_buffer;       /**< IceT depth buffer */

  IceTCommunicator comm;  /**< IceT MPI communicator */
  IceTInt *process_ranks; /**< IceT Internal ranks */
  IceTInt proc;           /**< IceT mpi process id */

  /**
   * @brief constructor
   * @param width Buffer width
   * @param height Buffer height
   */
  IceTComposite(std::size_t width = 0, std::size_t height = 0);

  ~IceTComposite();

  /**
   * Reset the buffer (e.g. set all values to 0)
   * @method reset
   */
  virtual void reset();
  /**
   * Call the composite method when several compute nodes cooperate to produce the final image
   * @method composite
   * @return [description]
   */
  virtual float *composite();
  /**
   * Add contribution to the buffer
   * @method localAdd
   * @param  x        x coordinate
   * @param  y        y coordinate
   * @param  color    color contribution
   * @param  alpha    alpha value
   * @param  t        depth value
   */
  virtual void localAdd(size_t x, size_t y, const glm::vec3 &color, float alpha = 1.f, float t = 0.f);
  /**
   * Add contribution to the buffer
   * @method localAdd
   * @param  i buffer index  = floor (y / width) + (x mod width)
   * @param  color    color contribution
   * @param  alpha    alpha value
   * @param  t        depth value
   */
  virtual void localAdd(size_t i, const glm::vec3 &color, float alpha = 1.f, float t = 0.f);
  /**
   * Write final color buffer to disk as a ppm image
   * @method write
   * @param  filename Filename to use
   */
  virtual void write(std::string filename);

  /**
   * Returns final buffer after @composite
   * @method colorbf
   * @return pointer to the final color buffer (nullptr if composite was not invoked)
   */
  virtual float *colorbf() { return color_buffer_final; }
  /**
   * Local depth buffer
   * @method depthbf
   * @return Pointer to the local depth buffer
   */
  virtual float *depthbf() { return depth_buffer; }
};
}
}
}
#endif /* COMPOSITE_H */
