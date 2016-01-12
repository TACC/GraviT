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
#include <gvt/render/data/scene/gvtCamera.h>
#include <boost/timer/timer.hpp>

using namespace gvt::core::math;
using namespace gvt::render::data::scene;
using namespace gvt::render::actor;

// Camera base class methods
gvtCameraBase::gvtCameraBase() {
  eye_point = Point4f(0, 0, 0, 1);
  focal_point = Point4f(0, 0, 0, 1);
  up_vector = Vector4f(0, 1, 0, 0);
  view_direction = Vector4f(0, 0, -1, 0);
  filmsize[0] = 512;
  filmsize[1] = 512;
  u = Vector4f(1, 0, 0, 0);
  v = Vector4f(0, 1, 0, 0);
  w = Vector4f(0, 0, 1, 0);
  cam2wrld = AffineTransformMatrix<float>(true);
  wrld2cam = AffineTransformMatrix<float>(true);
  INVRAND_MAX = 1.0 / (float)RAND_MAX;
}
gvtCameraBase::gvtCameraBase(const gvtCameraBase &cam) {
  eye_point = cam.eye_point;
  focal_point = cam.focal_point;
  up_vector = cam.up_vector;
  view_direction = cam.view_direction;
  filmsize[0] = cam.filmsize[0];
  filmsize[1] = cam.filmsize[1];
}
float gvtCameraBase::frand() { return ((float)rand()) * INVRAND_MAX; }
void gvtCameraBase::SetCamera(gvt::render::actor::RayVector &rayvect, float rate) {
  rays = rayvect;
  rate = rate;
}
void gvtCameraBase::buildTransform() {
  //
  // Find the u, v, and w unit basis vectors for the camera coordinate system.
  // These vectors are in world coordinates and are considered attached to the camera.
  // Calculate the w vector that points from the camera to the focal point.
  // Normalize it.
  //
  w = focal_point - eye_point;
  w.normalize();
  //
  // V direction is the camera up vector.
  //
  v = up_vector.normalize();
  //
  // U direction is the cross product of the camera up vector and the W vector
  // (left hand camera coord system)
  //
  u[0] = v[1] * w[2] - v[2] * w[1];
  u[1] = v[2] * w[0] - v[0] * w[2];
  u[2] = v[0] * w[1] - v[1] * w[0];
  u[3] = 0.0;
  u = u.normalize();
  //
  // The up vector input may not have been orthogonal to the viewing direction w.
  // Recalculate an up vector perpendicular to both the view direction w and the
  // horizontal direction up. The new up vector will be the cross product of w and u
  //
  up_vector[0] = w[1] * u[2] - u[1] * w[2];
  up_vector[1] = w[2] * u[0] - u[2] * w[0];
  up_vector[2] = w[0] * u[1] - u[0] * w[1];
  up_vector[3] = 0.0;
  v = up_vector.normalize();
  //
  // last column in the camera to world transformation matrix is the eye_point.
  //
  cam2wrld.n[3] = eye_point[0];
  cam2wrld.n[7] = eye_point[1];
  cam2wrld.n[11] = eye_point[2];
  cam2wrld.n[15] = eye_point[3];
  //
  // Third column in the camera to world transformation matrix contains the
  // unit vector from the eye_point to the focal_point.
  //
  cam2wrld.n[2] = w[0];
  cam2wrld.n[6] = w[1];
  cam2wrld.n[10] = w[2];
  cam2wrld.n[14] = w[3];
  //
  cam2wrld.n[1] = v[0];
  cam2wrld.n[5] = v[1];
  cam2wrld.n[9] = v[2];
  cam2wrld.n[13] = v[3];
  //
  cam2wrld.n[0] = u[0];
  cam2wrld.n[4] = u[1];
  cam2wrld.n[8] = u[2];
  cam2wrld.n[12] = u[3];
  //
  // invert for world to camera transform
  //
  wrld2cam = cam2wrld.inverse();
}

void gvtCameraBase::setFilmsize(const int film_size[]) {
  filmsize[0] = film_size[0];
  filmsize[1] = film_size[1];
}
void gvtCameraBase::setFilmsize(int width, int height) {
  filmsize[0] = width;
  filmsize[1] = height;
}
int gvtCameraBase::getFilmSizeWidth() { return filmsize[0]; }
int gvtCameraBase::getFilmSizeHeight() { return filmsize[1]; }
void gvtCameraBase::setEye(const Vector4f &eye) { eye_point = eye; }
void gvtCameraBase::lookAt(Point4f eye, Point4f focus, Vector4f up) {
  eye_point = eye;
  focal_point = focus;
  view_direction = focal_point - eye_point;
  up_vector = up;
  //
  buildTransform();
}
// gvt::render::actor::RayVector gvtCameraBase::AllocateCameraRays() {
void gvtCameraBase::AllocateCameraRays() {
#ifdef GVT_USE_DEBUG
  boost::timer::auto_cpu_timer t("gvtCameraBase::AllocateCameraRays: time: %w\n");
#endif
  depth = 0;
  size_t nrays = filmsize[0] * filmsize[1];
  rays.resize(nrays);
  // return rays;
}
gvtCameraBase::~gvtCameraBase() {}

// Perspective camera methods
gvtPerspectiveCamera::gvtPerspectiveCamera() { field_of_view = 30.0; }
gvtPerspectiveCamera::gvtPerspectiveCamera(const gvtPerspectiveCamera &cam) : gvtCameraBase(cam) {
  field_of_view = cam.field_of_view;
}
gvtPerspectiveCamera::~gvtPerspectiveCamera() {}
// gvt::render::actor::RayVector gvtPerspectiveCamera::generateRays() {
void gvtPerspectiveCamera::generateRays() {
#ifdef GVT_USE_DEBUG
  boost::timer::auto_cpu_timer t("gvtPerspectiveCamera::generateRays: time: %w\n");
#endif
  // Generate rays direction in camera space and transform to world space.
  int buffer_width = filmsize[0];
  int buffer_height = filmsize[1];
  int i, j, idx;
  float aspectRatio = float(buffer_width) / float(buffer_height);
  float x, y;
  // these basis directions are scaled by the aspect ratio and
  // the field of view.
  Vector4f camera_vert_basis_vector = Vector4f(0, 1, 0, 0) * tan(field_of_view * 0.5);
  Vector4f camera_horiz_basis_vector = Vector4f(1, 0, 0, 0) * tan(field_of_view * 0.5) * aspectRatio;
  Vector4f camera_normal_basis_vector = Vector4f(0, 0, 1, 0);
  Vector4f camera_space_ray_direction;
  for (j = 0; j < buffer_height; j++)
    for (i = 0; i < buffer_width; i++) {
      // select a ray and load it up
      idx = j * buffer_width + i;
      Ray &ray = rays[idx];
      ray.id = idx;
      ray.w = 1.0; // ray weight 1 for no subsamples. mod later
      ray.origin = eye_point;
      ray.type = Ray::PRIMARY;
      // calculate scale factors -1.0 < x,y < 1.0
      x = 2.0 * float(i) / float(buffer_width - 1) - 1.0;
      y = 2.0 * float(j) / float(buffer_height - 1) - 1.0;
      // calculate ray direction in camera space;
      camera_space_ray_direction =
          camera_normal_basis_vector + x * camera_horiz_basis_vector + y * camera_vert_basis_vector;
      // transform ray to world coordinate space;
      ray.setDirection(cam2wrld * camera_space_ray_direction.normalize());
      ray.depth = depth;
    }
}
void gvtPerspectiveCamera::setFOV(const float fov) { field_of_view = fov; }
