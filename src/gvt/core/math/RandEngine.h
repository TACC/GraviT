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

#ifndef GVT_CORE_MATH_RAND_ENGINE_H
#define GVT_CORE_MATH_RAND_ENGINE_H

namespace gvt {
namespace core {
namespace math {

class RandEngine {
public:
#define rotl(r, n) (((r) << (n)) | ((r) >> ((8 * sizeof(r)) - (n))))

  inline float rng(uint &seed) {
    uint x, y, z;
    x = (seed >> 16) + 4125832013u;   // upper 16 bits + offset
    y = (seed & 0xffff) + 814584116u; // lower 16 bits + offset
    z = 542;
    x *= 255519323u;
    x = rotl(x, 13); // CMR, period = 4294785923 (prime)
    y *= 3166389663u;
    y = rotl(y, 17); // CMR, period = 4294315741 (prime)
    z -= rotl(z, 11);
    z = rotl(z, 27); // RSR, period = 253691 = 2^3*3^2*71*557
    seed = x ^ y ^ z;
    return ((float)(seed & 0x00FFFFFF) / (float)0x01000000);
  }

  inline float rnghost(uint &seed) {
    uint x, y, z;
    x = (seed >> 16) + 4125832013u;   // upper 16 bits + offset
    y = (seed & 0xffff) + 814584116u; // lower 16 bits + offset
    z = 542;
    x *= 255519323u;
    x = rotl(x, 13); // CMR, period = 4294785923 (prime)
    y *= 3166389663u;
    y = rotl(y, 17); // CMR, period = 4294315741 (prime)
    z -= rotl(z, 11);
    z = rotl(z, 27); // RSR, period = 253691 = 2^3*3^2*71*557
    seed = x ^ y ^ z;
    return ((float)(seed & 0x00FFFFFF) / (float)0x01000000);
  }

  inline float fastrand(float min, float max) {
    // g_seed = 214013 * (g_seed) + 2531011;
    return min + rng(g_seed) * (max - min);
  }

  inline float fastrand(unsigned int *seedval, float min, float max) {
    *seedval = 214013 * (*seedval) + 2531011;
    return min + (*seedval >> 16) * ff * (max - min);
  }

  void SetSeed(unsigned int seedval) { g_seed = seedval; }

  unsigned int *ReturnSeed() { return &g_seed; }

protected:
  unsigned int g_seed;
  const float ff = (1.0f / 65535.0f);
};
}
}
}

#endif
