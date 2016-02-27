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
#include <gvt/render/data/scene/Camera.h>

#include <gvt/core/Math.h>

using namespace gvt::core::math;
using namespace gvt::render::actor;
using namespace gvt::render::data::scene;
using namespace std;

void Camera::SetCamera(RayVector &rays, float rate) {
  this->rays = rays;
  this->rate = rate;
  this->trcUpSampling = 1;
}

struct cameraGenerateRays {
  Camera *cam;
  size_t start, end;

  cameraGenerateRays(Camera *cam, size_t start, size_t end) : cam(cam), start(start), end(end) {}

  inline float frand() { return (((float)rand() / RAND_MAX) - 0.5f) * 2.0f; }

  void operator()() {
    AffineTransformMatrix<float> m = cam->m; // rotation matrix
    int depth = cam->depth;
    RayVector &rays = cam->rays;
    Vector4f eye = cam->eye;
    Vector4f look = cam->look;       // direction to look
    Vector4f u = cam->u, v = cam->v; // u and v in the

    const float divider = cam->trcUpSampling;
    const float offset = 1.0 / divider;
    const float offset2 = offset / 2.f;
    const float w = 1.0 / (divider * divider);
    const float buffer_width = cam->getFilmSizeWidth();
    const float buffer_height = cam->getFilmSizeHeight();
    Vector4f dir;
    for (size_t j = start; j < end; j++) {
      for (int i = 0; i < buffer_width; i++) {
        int idx = j * buffer_width + i;
        for (float off_i = 0; off_i < 1.0; off_i += offset) {
          for (float off_j = 0; off_j < 1.0; off_j += offset) {
            float x1 = float(i) + off_i + offset2 * (frand() - 0.5);
            float y1 = float(j) + off_j + offset2 * (frand() - 0.5);
            float x = x1 / float(buffer_width) - 0.5;
            float y = y1 / float(buffer_height) - 0.5;

            dir = m * ((look + x * u + y * v)).normalize();

            Ray &ray = rays[idx];
            ray.id = idx;
            ;
            ray.origin = eye;
            ray.w = w;
            ray.depth = depth;
            ray.setDirection(dir);
            ray.type = Ray::PRIMARY;
          }
        }
      }
    }
  }
};

RayVector &Camera::MakeCameraRays() {

  // trcUpSampling = 1;
  // depth = 0;
  // size_t nrays = (trcUpSampling * trcUpSampling) * filmsize[0] * filmsize[1];
  // int offset = filmsize[1] / gvt::core::schedule::asyncExec::instance()->numThreads;
  // {
  //   boost::timer::auto_cpu_timer t("Allocate camera rays %t\n");
  //   rays.resize(nrays);
  // }

  // {
  //   boost::timer::auto_cpu_timer t("Generating camera rays %t\n");
  //   cameraGenerateRays(this, 0, filmsize[1])();
  // }

  // GVT_DEBUG(DBG_ALWAYS, "EXPECTED PREGENERATING : " << nrays);
  // GVT_DEBUG(DBG_ALWAYS, "PREGENERATING : " << rays.size());
  return rays;
}
