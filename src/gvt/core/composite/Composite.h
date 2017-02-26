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

class AbstractCompositeBuffer {
public:
  AbstractCompositeBuffer() {}
  ~AbstractCompositeBuffer() {}

  virtual void reset(){};
};

template <typename T> class Buffer : public AbstractCompositeBuffer {
protected:
  std::size_t width, height;

public:
  Buffer(std::size_t width = 0, std::size_t height = 0)
      : AbstractCompositeBuffer(), width(width), height(height) {}
  ~Buffer() {}
  virtual void reset(){};
  virtual T *composite() { return nullptr; };
  virtual void localAdd(size_t x, size_t y, const glm::vec3 &color, float alpha = 1.f,
                        float t = 0.f){};
  virtual void localAdd(size_t i, const glm::vec3 &color, float alpha = 1.f,
                        float t = 0.f){};

private:
};
}
}
}
#endif /* COMPOSITE_H */
