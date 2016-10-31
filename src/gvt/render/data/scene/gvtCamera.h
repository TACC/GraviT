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
#ifndef GVT_RENDER_DATA_SCENE_GVTCAMERA_H
#define GVT_RENDER_DATA_SCENE_GVTCAMERA_H

// this file contains definitions for cameras used in GraviT.
//
// Dave Semeraro - May 2015
//
#include <gvt/core/Math.h>
#include <gvt/core/math/RandEngine.h>
#include <gvt/render/data/Primitives.h>
#include <stdlib.h>

namespace gvt {
namespace render {
namespace data {
namespace scene {

#define RIGHT_HAND_CAMERA // OpenGL Standard
                          //#define LEFT_HAND_CAMERA

/// gvtCameraBase - A base class for GraviT cameras.
/** The base class contains all the methods and variables common
 *  to all cameras in GraviT. The camera class maintains all the camera state. It
 *  also contains the vector of primary rays. An affine transformation matrix is
 *  maintained that transforms from camera to world and from world to camera coordinate
 *  systems. Left handed coordinate systems are assumed for both. */

class gvtCameraBase {
public:
  /** Default constructor sets eye_point and focal_point to ( 0, 0, 0, 1),
   *  and view direction too (0, 0, -1, 0) (looking down z axis). The up
   *  vector is set to (0, 1, 0, 0) This is the OpenGL camera default.
   */
  gvtCameraBase();

  /** Copy constructor */
  gvtCameraBase(const gvtCameraBase &cam);

  /** Destructor */
  ~gvtCameraBase();

  /** Set film size film_size[0] is width */
  void setFilmsize(const int film_size[]);

  /** Set Film size from two ints */
  void setFilmsize(int width, int height);

  /** get the film width */
  int getFilmSizeWidth();

  /** get the film height */
  int getFilmSizeHeight();

  /** Set the eye point or position of the camera. This call causes recomputation of the
   *  transformation matrix, and all other camera parameters impacted by the change. */
  void setEye(const glm::vec3 &eye);

  /** Pass in the camera ray vector reference so the camera can populate it with rays.
   *  This method is not really necessary and is only here for backward compatibility. */
  void SetCamera(gvt::render::actor::RayVector &rays, float rate);

  /** Return a random floating point value between 0.0 and 1.0 */
  float frand();

  /** Fill up the ray vector with correct rays. Base class just initializes the vector.
   *  Derived classes insert the rays themselves. */
  virtual void AllocateCameraRays();

  /** given a new eye point, focal point, and up vector modify all the other dependant vars.
   *  in particular rebuild the transformation matrix. The new camera position is passed in as
   *  eye. The new focal point is passed in as focus. And the camera up vector is passed in as up. The
   *  camera coordinate system with unit vectors, u, v, and w is constructed. From this the camera
   *  to world transformation and its inverse are constructed. */
  void lookAt(glm::vec3 eye, glm::vec3 focus, glm::vec3 up);

  void setSamples(int pathSamples);

  void setMaxDepth(int depth);

  void setJitterWindowSize(int windowSize);

  /** Bunch-o-rays */
  gvt::render::actor::RayVector rays;

  // clang-format off
  glm::vec3 getEyePoint() {
    return eye_point;
  }

  glm::vec3 getFocalPoint() {
    return focal_point;
  }

  glm::vec3 getUpVector() {
    return up_vector;
  }

  // clang-format on
public:
  int samples;
  int jitterWindowSize;
  glm::mat4 cam2wrld;       //!< transform from camera to world coords
  glm::mat4 wrld2cam;       //!< transform from world to camera coords
  glm::vec3 eye_point;      //!< camera location in world coordinates
  glm::vec3 focal_point;    //!< camera focal point in world coordinates
  glm::vec3 up_vector;      //!< vector pointing "up".
  glm::vec3 view_direction; //!< direction camera is looking. generally focal - eye pt.
  int filmsize[2];          //!< image size dimensions in pixels. filmsize[0] = width.
  int depth;                //!< legacy variable from previous cameras. Initializes ray depth
  glm::vec3 u, v, w;        //!< unit basis vectors for camera space in world coords.
  float INVRAND_MAX;
  gvt::core::math::RandEngine randEngine;
  //
  void buildTransform(); //!< Build the transformation matrix and inverse
};

/// gvtPerspectiveCamera - a camera that produces a perspective view
/** This camera produces a perspective view of the world. The field of view is the
 *  angle subtended by the film plane width from the eye point. This class has methods
 *  to allocate rays and to initialize or generate the primary ray set based on the
 *  camera state.
 */
class gvtPerspectiveCamera : public gvtCameraBase {
public:
  /** Perspective camera default constructor */
  gvtPerspectiveCamera();

  /** Perspective camera copy constructor */
  gvtPerspectiveCamera(const gvtPerspectiveCamera &cam);

  /** Perspective camera destructor */
  ~gvtPerspectiveCamera();

  /** Set the field of view angle in degrees*/
  void setFOV(const float fov);

  /** Fill the ray data structure */
  virtual void generateRays();

protected:
  float field_of_view; //!< Angle subtended by the film plane height from eye_point
};

} // scene
} // data
} // render
} // gvt

#endif // GVT_RENDER_DATA_SCENE_GVTCAMERA_H
