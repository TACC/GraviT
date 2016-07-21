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
//
// Image.h
//

#ifndef GVT_RENDER_DATA_SCENE_IMAGE_H
#define GVT_RENDER_DATA_SCENE_IMAGE_H

#include <glm/glm.hpp>
#include <gvt/render/data/scene/ColorAccumulator.h>

#include <string>

namespace gvt {
namespace render {
namespace data {
namespace scene {
/// image buffer
/** image buffer used to accumulate the final image
*/
class Image {
public:
  // clang-format off
  enum ImageFormat {
    PPM
  };
  // clang-format on

  Image(int w, int h, std::string fn = "gvt_image", ImageFormat f = PPM)
      : width(w), height(h), filename(fn), format(f) {
    int size = 3 * width * height;
    rgb = new unsigned char[size];
    for (int i = 0; i < size; ++i) rgb[i] = 0;
  }

  void Add(int pixel, float *buf) {
    int index = 3 * pixel;
    rgb[index + 0] = (unsigned char)(buf[0] * 256.f);
    rgb[index + 1] = (unsigned char)(buf[1] * 256.f);
    rgb[index + 2] = (unsigned char)(buf[2] * 256.f);
  }

  void Add(int pixel, glm::vec3 &ca) {
    int index = 3 * pixel;
    rgb[index + 0] = (unsigned char)(ca[0] * 255.f);
    rgb[index + 1] = (unsigned char)(ca[1] * 255.f);
    rgb[index + 2] = (unsigned char)(ca[2] * 255.f);
    if (rgb[index + 0] > 255.f) rgb[index + 0] = 255;
    if (rgb[index + 1] > 255.f) rgb[index + 1] = 255;
    if (rgb[index + 2] > 255.f) rgb[index + 2] = 255;
  }

  void Add(int pixel, glm::vec3 &ca, float w) {
    int index = 3 * pixel;
    rgb[index + 0] = ((unsigned char)(ca[0] * 255.f) * w);
    rgb[index + 1] = ((unsigned char)(ca[1] * 255.f) * w);
    rgb[index + 2] = ((unsigned char)(ca[2] * 255.f) * w);
  }

  unsigned char *GetBuffer() { return rgb; }

  void Write();
  void resize(const size_t &w, const size_t &h) {
    if (rgb) delete[] rgb;
    width = w;
    height = h;
    int size = 3 * width * height;
    rgb = new unsigned char[size];
    std::memset(rgb, 0, sizeof(char) * 3 * width * height);
  }

  void clear() { std::memset(rgb, 0, sizeof(char) * 3 * width * height); }

  ~Image() { delete[] rgb; }

private:
  int width, height;
  std::string filename;
  ImageFormat format;
  unsigned char *rgb;
};
}
}
}
}

#endif // GVT_RENDER_DATA_SCENE_IMAGE_H
