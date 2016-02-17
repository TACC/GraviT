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
#ifndef GVT_RENDER_DATA_SCENE_CAMERA_H
#define GVT_RENDER_DATA_SCENE_CAMERA_H

#include <gvt/core/Math.h>
#include <gvt/core/schedule/TaskScheduling.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/CameraConfig.h>

#include <boost/aligned_storage.hpp>
#include <boost/foreach.hpp>
#include <boost/timer/timer.hpp>

#include <time.h>

namespace gvt {
namespace render {
namespace data {
namespace scene {
/// viewpoint through which the scene is rendered
class Camera {
public:
  Camera() {
    aspectRatio = 1;
    normalizedHeight = 1;
    eye = gvt::core::math::Vector4f(0, 0, 0, 1);
    u = gvt::core::math::Vector4f(1, 0, 0, 0);
    v = gvt::core::math::Vector4f(0, 1, 0, 0);
    look = gvt::core::math::Vector4f(0, 0, -1, 1);
    focus = gvt::core::math::Vector4f(0, 0, -1, 1);
  }

  void SetCamera(gvt::render::actor::RayVector &rays, float rate);

  void setFilmSize(int width, int height) {
    filmsize[0] = width;
    filmsize[1] = height;

    setAspectRatio(double(width) / double(height));
  }

  float getFilmSizeWidth(void) { return filmsize[0]; }

  float getFilmSizeHeight(void) { return filmsize[1]; }

  void setEye(const gvt::core::math::Vector4f &eye) { this->eye = eye; }

  void setLook(double r, double i, double j, double k) {
    // set look matrix
    m[0][0] = 1.0 - 2.0 * (i * i + j * j);
    m[0][1] = 2.0 * (r * i - j * k);
    m[0][2] = 2.0 * (j * r + i * k);

    m[1][0] = 2.0 * (r * i + j * k);
    m[1][1] = 1.0 - 2.0 * (j * j + r * r);
    m[1][2] = 2.0 * (i * j - r * k);

    m[2][0] = 2.0 * (j * r - i * k);
    m[2][1] = 2.0 * (i * j + r * k);
    m[2][2] = 1.0 - 2.0 * (i * i + r * r);

    update();
  }

  void setLook(const gvt::core::math::Vector4f &viewDir, const gvt::core::math::Vector4f &upDir) {
    gvt::core::math::Vector3f z = viewDir;      // this is where the z axis should end up
    const gvt::core::math::Vector3f &y = upDir; // where the y axis should end up
    gvt::core::math::Vector3f x = y ^ z;        // lah,
    // clang-format off
    m = gvt::core::math::AffineTransformMatrix<float>(x[0], x[1], x[2], 0.f, y[0], y[1], y[2], 0.f, z[0], z[1], z[2],
                                                      0.f, 0.f, 0.f, 0.f, 1.f).transpose();
    // clang-format on
    update();
  }

  void setLook(gvt::core::math::Vector4f &eyePos, gvt::core::math::Vector4f &lookAt,
               const gvt::core::math::Vector4f &upDir) {
    eye = eyePos;
    look = lookAt;
    up = upDir;
    focus = lookAt;
    gvt::core::math::Vector3f z = -(lookAt - eyePos).normalize(); // this is where the z axis should end up
    const gvt::core::math::Vector3f y = upDir;                    // where the y axis should end up
    gvt::core::math::Vector3f x = (y ^ z).normalize();

    // clang-format off
    m = gvt::core::math::AffineTransformMatrix<float>(x[0], x[1], x[2], 0.f, y[0], y[1], y[2], 0.f, z[0], z[1], z[2],
                                                      0.f, 0.f, 0.f, 0.f, 1.f).transpose();
    // clang-format on
    update();
    // const gvt::core::math::AffineTransformMatrix<float> minv = m.inverse();
  }

  void setFOV(double fov) {
    normalizedHeight = tan(fov / 2.0);
    update();
  }

  void setAspectRatio(double ar) {
    aspectRatio = ar;
    update();
  }

  double getAspectRatio() { return aspectRatio; }

  const gvt::core::math::Vector4f &getEye() const { return eye; }

  const gvt::core::math::Vector4f &getFocus() const { return focus; }
  const gvt::core::math::Vector4f &getLook() const { return look; }

  const gvt::core::math::Vector4f &getU() const { return u; }

  const gvt::core::math::Vector4f &getV() const { return v; }

  const gvt::core::math::Vector4f &getUp() const { return up; }

  const gvt::core::math::AffineTransformMatrix<float> getMatrix() { return m; }

  float frand() { return .1f; }

  double gauss(double x) { return 0.5 * exp(-((x - 1.0) * (x - 1.0)) / 0.2); }

  gvt::render::actor::RayVector &MakeCameraRays();

  boost::mutex rmutex;

  gvt::core::math::AffineTransformMatrix<float> m; // rotation matrix
  double normalizedHeight;                         // dimensions of image place at unit dist from eye
  double aspectRatio;

  void update() {
    // using the above three values calculate look,u,v
    u = m * gvt::core::math::Vector3f(1, 0, 0) * normalizedHeight * aspectRatio;
    v = m * gvt::core::math::Vector3f(0, 1, 0) * normalizedHeight;
    look = gvt::core::math::Vector3f(0, 0, -1);
  }

  gvt::core::math::Vector4f eye;
  gvt::core::math::Vector4f look;  // direction to look
  gvt::core::math::Vector4f focus; // focal point
  gvt::core::math::Vector4f up;    // direction to look
  gvt::core::math::Vector4f u, v;  // u and v in the

  gvt::render::actor::RayVector rays;
  float rate;
  int trcUpSampling;
  int depth;
  int filmsize[2];
};
}
}
}
}
#endif // GVT_RENDER_DATA_SCENE_CAMERA_H
