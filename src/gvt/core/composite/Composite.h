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

#ifndef GVT_CORE_COMPOSITE_H
#define GVT_CORE_COMPOSITE_H

#include <cstddef>
#include <glm/glm.hpp>

namespace gvt {
namespace core {
namespace composite {

/**
 * @brief Abstract Composite Buffer
 *
 * Abtract generic class that all Composite buffers derive from. Use for polimorphism.
 *
 */

class AbstractCompositeBuffer {
public:
  AbstractCompositeBuffer() {}
  ~AbstractCompositeBuffer() {}

  virtual void reset(){};
};
/**
 * @brief Buffer
 *
 * Implements/defines all the required methods required by the scheduler to composite the image at the end of the frame.
 *
 * T defines the type of data (e.g. float RGBA requires a struct with 4 floats)
 *
 */
template <typename T> class Buffer : public AbstractCompositeBuffer {
protected:
  std::size_t width /**< Width of the image */, height /**< Height of the image */;

public:
  /**
   * @brief constructor
   *
   * @param width Width of the 2D buffer
   * @param height Height of the 2D buffers
   */
  Buffer(std::size_t width = 0, std::size_t height = 0) : AbstractCompositeBuffer(), width(width), height(height) {}
  ~Buffer() {}
  /**
   * Reset the buffer (e.g. set all values to 0)
   * @method reset
   */
  virtual void reset(){};
  /**
   * Call the composite method when several compute nodes cooperate to produce the final image
   * @method composite
   * @return [description]
   */
  virtual T *composite() { return nullptr; };
  /**
   * Add contribution to the buffer
   * @method localAdd
   * @param  x        x coordinate
   * @param  y        y coordinate
   * @param  color    color contribution
   * @param  alpha    alpha value
   * @param  t        depth value
   */
  virtual void localAdd(size_t x, size_t y, const glm::vec3 &color, float alpha = 1.f, float t = 0.f){};
  /**
   * Add contribution to the buffer
   * @method localAdd
   * @param  i buffer index  = floor (y / width) + (x mod width)
   * @param  color    color contribution
   * @param  alpha    alpha value
   * @param  t        depth value
   */
  virtual void localAdd(size_t i, const glm::vec3 &color, float alpha = 1.f, float t = 0.f){};

private:
};
}
}
}
#endif /* COMPOSITE_H */
