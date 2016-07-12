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
#include <boost/timer/timer.hpp>

#include <gvt/core/utils/timer.h>

#include <gvt/render/data/scene/gvtCamera.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <thread>

using namespace gvt::render::data::scene;
using namespace gvt::render::actor;

// Camera base class methods
gvtCameraBase::gvtCameraBase() {
  eye_point = glm::vec3(0, 0, 0);
  focal_point = glm::vec3(0, 0, 0);
  up_vector = glm::vec3(0, 1, 0);
  view_direction = glm::vec3(0, 0, -1);
  filmsize[0] = 512;
  filmsize[1] = 512;
  u = glm::vec3(1, 0, 0);
  v = glm::vec3(0, 1, 0);
  w = glm::vec3(0, 0, 1);
  cam2wrld = glm::mat4(1.f);
  wrld2cam = glm::mat4(1.f);
  INVRAND_MAX = 1.0 / (float)RAND_MAX;
  jitterWindowSize = 0.000;
  samples = 1;
  depth = 1;
}
gvtCameraBase::gvtCameraBase(const gvtCameraBase &cam) {
  eye_point = cam.eye_point;
  focal_point = cam.focal_point;
  up_vector = cam.up_vector;
  view_direction = cam.view_direction;
  filmsize[0] = cam.filmsize[0];
  filmsize[1] = cam.filmsize[1];
  jitterWindowSize = cam.jitterWindowSize;
  samples = cam.samples;
  depth = cam.depth;
}
float gvtCameraBase::frand() { return ((float)rand()) * INVRAND_MAX; }
void gvtCameraBase::SetCamera(gvt::render::actor::RayVector &rayvect, float _rate) { rays = rayvect; }
void gvtCameraBase::buildTransform() {
  //
  // Find the u, v, and w unit basis vectors for the camera coordinate system.
  // These vectors are in world coordinates and are considered attached to the camera.
  // Calculate the w vector that points from the camera to the focal point.
  // Normalize it.
  //
  w = glm::normalize(focal_point - eye_point);
  //
  // V direction is the camera up vector.
  //
  v = glm::normalize(up_vector);
//
// U direction is the cross product of the camera up vector and the W vector
//
//
#ifdef LEFT_HAND_CAMERA
  u[0] = v[1] * w[2] - v[2] * w[1];
  u[1] = v[2] * w[0] - v[0] * w[2];
  u[2] = v[0] * w[1] - v[1] * w[0];
  u = glm::normalize(u);

#endif

#ifdef RIGHT_HAND_CAMERA
  u[0] = w[1] * v[2] - w[2] * v[1];
  u[1] = w[2] * v[0] - w[0] * v[2];
  u[2] = w[0] * v[1] - w[1] * v[0];

  u = glm::normalize(u);
#endif

//
// The up vector input may not have been orthogonal to the viewing direction w.
// Recalculate an up vector perpendicular to both the view direction w and the
// horizontal direction up. The new up vector will be the cross product of w and u
//
#ifdef LEFT_HAND_CAMERA
  up_vector[0] = w[1] * u[2] - u[1] * w[2];
  up_vector[1] = w[2] * u[0] - u[2] * w[0];
  up_vector[2] = w[0] * u[1] - u[0] * w[1];
  // up_vector[3] = 0.0;
  v = glm::normalize(up_vector);
#endif

#ifdef RIGHT_HAND_CAMERA
  up_vector[0] = u[1] * w[2] - w[1] * u[2];
  up_vector[1] = u[2] * w[0] - w[2] * u[0];
  up_vector[2] = u[0] * w[1] - w[0] * u[1];
  // up_vector[3] = 0.0;
  v = glm::normalize(up_vector);
#endif

  //
  // last column in the camera to world transformation matrix is the eye_point.
  //
  cam2wrld[0][3] = eye_point[0];
  cam2wrld[1][3] = eye_point[1];
  cam2wrld[2][3] = eye_point[2];
  cam2wrld[3][3] = 1.f; // eye_point[3];
  //
  // Third column in the camera to world transformation matrix contains the
  // unit vector from the eye_point to the focal_point.
  //
  cam2wrld[0][2] = w[0];
  cam2wrld[1][2] = w[1];
  cam2wrld[2][2] = w[2];
  cam2wrld[3][2] = 0.f;
  //
  cam2wrld[0][1] = v[0];
  cam2wrld[1][1] = v[1];
  cam2wrld[2][1] = v[2];
  cam2wrld[3][1] = 0.f;
  //
  cam2wrld[0][0] = u[0];
  cam2wrld[1][0] = u[1];
  cam2wrld[2][0] = u[2];
  cam2wrld[3][0] = 0.f;
  //
  // invert for world to camera transform
  //
  wrld2cam = glm::inverse(cam2wrld);
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
void gvtCameraBase::setEye(const glm::vec3 &eye) { eye_point = eye; }
void gvtCameraBase::lookAt(glm::vec3 eye, glm::vec3 focus, glm::vec3 up) {
  eye_point = eye;
  focal_point = focus;
  view_direction = focal_point - eye_point;
  up_vector = up;
  //
  buildTransform();
}

void gvtCameraBase::setSamples(int pathSamples) { samples = pathSamples; }

void gvtCameraBase::setMaxDepth(int maxRayDepth) { depth = maxRayDepth; }

void gvtCameraBase::setJitterWindowSize(int windowSize) { jitterWindowSize = windowSize; }

// gvt::render::actor::RayVector gvtCameraBase::AllocateCameraRays() {
void gvtCameraBase::AllocateCameraRays() {
#ifdef GVT_USE_DEBUG
  boost::timer::auto_cpu_timer t("gvtCameraBase::AllocateCameraRays: time: %w\n");
#endif
  size_t nrays = filmsize[0] * filmsize[1] * samples * samples;
  rays.clear();
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
  //boost::timer::auto_cpu_timer t("gvtPerspectiveCamera::generateRays: time: %w\n");
#endif
  gvt::core::time::timer t(true, "generate camera rays");
  // Generate rays direction in camera space and transform to world space.
  int buffer_width = filmsize[0];
  int buffer_height = filmsize[1];

  float aspectRatio = float(buffer_width) / float(buffer_height);

  // these basis directions are scaled by the aspect ratio and
  // the field of view.

  const float vert = tanf(field_of_view * 0.5);
  const float horz = tanf(field_of_view * 0.5) * aspectRatio;

  const glm::vec4 camera_normal_basis_vector(0, 0, 1, 0);

  const float divider = samples;
  const float offset = (1.0 / divider) * jitterWindowSize;
  const glm::vec3 z(cam2wrld[0][2], cam2wrld[1][2], cam2wrld[2][2]);

  const float wmult = 2.f / float(buffer_width - 1);
  const float hmult = 2.f / float(buffer_height - 1);
  const float half_sample = samples * 0.5f;
  const size_t samples2 = samples * samples;
  const float contri = 1.f / (samples * samples);
  // for (j = 0; j < buffer_height; j++)
  //   for (i = 0; i < buffer_width; i++) {

  const size_t chunksize = buffer_height / (std::thread::hardware_concurrency() * 4);
  static tbb::simple_partitioner ap;
  tbb::parallel_for(tbb::blocked_range<size_t>(0, buffer_height, chunksize),
                    [&](tbb::blocked_range<size_t> &chunk) {
                      RandEngine randEngine;
                      randEngine.SetSeed(chunk.begin());
                      for (size_t j = chunk.begin(); j < chunk.end(); j++) {
                        // multi - jittered samples
                        // int i = idx;
                        int idx = j * buffer_width;
                        for (size_t i = 0; i < buffer_width; i++) {
                          const float x0 = float(i) * wmult - 1.0, y0 = float(j) * hmult - 1.0;
                          float x, y;
                          // glm::vec4 camera_space_ray_direction;
                          for (int k = 0; k < samples; k++) {
                            for (int w = 0; w < samples; w++) {
                              // calculate scale factors -1.0 < x,y < 1.0
                              int ridx = idx * samples2 + k * samples + w;
                              x = x0 + (w - half_sample) * offset; // + offset * (randEngine.fastrand(0, 1) - 0.5);
                              x *= horz;
                              y = y0 + (k - half_sample) * offset; // + offset * (randEngine.fastrand(0, 1) - 0.5);
                              y *= vert;
                              glm::vec3 camera_space_ray_direction;
                              camera_space_ray_direction[0] = cam2wrld[0][0] * x + cam2wrld[0][1] * y + z[0];
                              camera_space_ray_direction[1] = cam2wrld[1][0] * x + cam2wrld[1][1] * y + z[1];
                              camera_space_ray_direction[2] = cam2wrld[2][0] * x + cam2wrld[2][1] * y + z[2];
                              Ray &ray = rays[ridx];
                              ray.id = idx;
                              ray.t_min = gvt::render::actor::Ray::RAY_EPSILON;
                              ray.t = ray.t_max = FLT_MAX;
                              ray.w = contri;
                              ray.origin = eye_point;
                              ray.type = Ray::PRIMARY;
                              ray.direction = glm::normalize(camera_space_ray_direction);
                              ray.depth = depth;
                            }
                          }
                          idx++;
                        }
                      }
                    },
                    ap);
}
void gvtPerspectiveCamera::setFOV(const float fov) { field_of_view = fov; }
