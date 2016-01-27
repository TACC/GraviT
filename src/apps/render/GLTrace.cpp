/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray
   tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas
   at Austin
   All rights reserved.

   Licensed under the BSD 3-Clause License, (the "License"); you may not use
   this file
   except in compliance with the License.
   A copy of the License is included with this software in the file LICENSE.
   If your copy does not contain the License, you may obtain a copy of the
   License at:

       http://opensource.org/licenses/BSD-3-Clause

   Unless required by applicable law or agreed to in writing, software
   distributed under
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY
   KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under
   limitations under the License.

   GraviT is funded in part by the US National Science Foundation under awards
   ACI-1339863,
   ACI-1339881 and ACI-1339840
   =======================================================================================
   */
/// OpenGL-based render window using GraviT
/**
 * GLTrace - GraviT rendering with render window display.
 * Modeled after MPITrace example program. Added glut window calls.
 *
 *  vglrun -n 4 -o 0 ./GLtrace ../data/bunny.conf
 *
 * left arrow key shifts camera left right arrow shifts camera right.
 * hit 'q' or esc to exit.
 *
 */

#include <mpi.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ConfigFileLoader.h"
#include <gvt/core/Math.h>
#include <gvt/core/mpi/Wrapper.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/Attributes.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/data/scene/Camera.h>
#include <gvt/render/algorithm/Tracers.h>
#include <gvt/render/adapter/manta/Wrapper.h>

#if defined(__APPLE__)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

typedef enum { BVH_RENDER_MODE, FILM_RENDER_MODE } render_mode;

#define ESCAPE 27
#define ROTATE_STEP .1
#define MOVE_STEP 0.1

using namespace std;
using namespace gvtapps::render;
using namespace gvt::render::adapter::manta::data::domain;
using namespace gvt::render::data::domain;
using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;
using namespace gvt::core::mpi;
using namespace gvt::core::math;
using namespace gvt::render;
using namespace gvt::render::data::primitives;

// global variables used by glut callbacks.
//
GLubyte *imagebuffer;
static GLint width;
static GLint height;

int opengl_rank;
int mpi_rank;
bool update = false;
//render_mode renderMode = BVH_RENDER_MODE;
render_mode renderMode = FILM_RENDER_MODE;
gvt::core::DBNodeH camNode;

static int mouseButton0 = 0;
static int mouseButton2 = 0;
static int mouseGrabLastX = 0;
static int mouseGrabLastY = 0;
static double lastMouseUpdate = 0.0;

gvt::render::algorithm::AbstractTrace *tracer;
gvtPerspectiveCamera mycamera;
Image* imageptr;

inline double WallClockTime() {
#if defined(__linux__) || defined(__APPLE__) || defined(__CYGWIN__) || \
  defined(__OpenBSD__) || defined(__FreeBSD__)
  struct timeval t;
  gettimeofday(&t, NULL);

  return t.tv_sec + t.tv_usec / 1000000.0;
#elif defined(WIN32)
  return GetTickCount() / 1000.0;
#else
#error "Unsupported Platform !!!"
#endif
}


/*
 * TODO: Multiple static values re-calculated over and over in these
 * camera manipulation routines
 */

void Translate(Point4f &eye, Point4f &focus, const float k) {
  Vector4f v = camNode["upVector"].value().toVector4f();

  Vector4f w = focus - eye;  // Cam direction
  Vector4f move_dir;         // Cross(w,v)

  move_dir[0] = w[1] * v[2] - w[2] * v[1];
  move_dir[1] = w[2] * v[0] - w[0] * v[2];
  move_dir[2] = w[0] * v[1] - w[1] * v[0];
  move_dir[3] = 0.0;
  move_dir = move_dir.normalize();

  Vector4f t = k * move_dir;
  eye += t;
  focus += t;
}

void TranslateLeft(Point4f &eye, Point4f &focus, const float k) {
  Translate(eye, focus, -k);
}

void TranslateRight(Point4f &eye, Point4f &focus, const float k) {
  Translate(eye, focus, k);
}

void TranslateForward(Point4f &eye, Point4f &focus, const float k) {
  Point4f t = k * (focus - eye);
  eye += t;
  focus += t;
}

void TranslateBackward(Point4f &eye, Point4f &focus, const float k) {
  Point4f t = -k * (focus - eye);
  eye += t;
  focus += t;
}

void Rotate(Point4f &eye, Point4f &focus, const float angle,
            const Point4f &axis) {
  Vector4f p = focus - eye;

  gvt::core::math::Vector4f t = angle * axis;

  gvt::core::math::AffineTransformMatrix<float> mAA;
  mAA = gvt::core::math::AffineTransformMatrix<float>::createRotation(
        t[0], 1.0, 0.0, 0.0) *
      gvt::core::math::AffineTransformMatrix<float>::createRotation(
        t[1], 0.0, 1.0, 0.0) *
      gvt::core::math::AffineTransformMatrix<float>::createRotation(t[2], 0.0,
      0.0, 1.0);
  // Rotate focus point
 // focus = eye + mAA * p;

  eye = focus + (mAA * (-p));
}

void RotateLeft(Point4f &eye, Point4f &focus, float angle) {
  Vector4f v = camNode["upVector"].value().toVector4f();

  Vector4f w = focus - eye;  // Cam direction
  Vector4f move_dir_x;       // Cross(w,v)
  Vector4f move_dir_y;       // Cross(move_dir_x,w)

  move_dir_x[0] = w[1] * v[2] - w[2] * v[1];
  move_dir_x[1] = w[2] * v[0] - w[0] * v[2];
  move_dir_x[2] = w[0] * v[1] - w[1] * v[0];
  move_dir_x[3] = 0.0;
  move_dir_x = move_dir_x.normalize();

  move_dir_y[0] = move_dir_x[1] * w[2] - move_dir_x[2] * w[1];
  move_dir_y[1] = move_dir_x[2] * w[0] - move_dir_x[0] * w[2];
  move_dir_y[2] = move_dir_x[0] * w[1] - move_dir_x[1] * w[0];
  move_dir_y[3] = 0.0;
  move_dir_y = move_dir_y.normalize();

  Rotate(eye, focus, angle, move_dir_y);
}

void RotateRight(Point4f &eye, Point4f &focus, float angle) {
  Vector4f v = camNode["upVector"].value().toVector4f();

  Vector4f w = focus - eye;  // Cam direction
  Vector4f move_dir_x;       // Cross(w,v)
  Vector4f move_dir_y;       // Cross(move_dir_x,w)

  move_dir_x[0] = w[1] * v[2] - w[2] * v[1];
  move_dir_x[1] = w[2] * v[0] - w[0] * v[2];
  move_dir_x[2] = w[0] * v[1] - w[1] * v[0];
  move_dir_x[3] = 0.0;
  move_dir_x = move_dir_x.normalize();

  move_dir_y[0] = move_dir_x[1] * w[2] - move_dir_x[2] * w[1];
  move_dir_y[1] = move_dir_x[2] * w[0] - move_dir_x[0] * w[2];
  move_dir_y[2] = move_dir_x[0] * w[1] - move_dir_x[1] * w[0];
  move_dir_y[3] = 0.0;
  move_dir_y = move_dir_y.normalize();

  Rotate(eye, focus, -angle, move_dir_y);
}

void RotateUp(Point4f &eye, Point4f &focus, float angle) {
  Vector4f v = camNode["upVector"].value().toVector4f();

  Vector4f w = focus - eye;  // Cam direction
  Vector4f move_dir;         // Cross(w,v)

  move_dir[0] = w[1] * v[2] - w[2] * v[1];
  move_dir[1] = w[2] * v[0] - w[0] * v[2];
  move_dir[2] = w[0] * v[1] - w[1] * v[0];
  move_dir[3] = 0.0;
  move_dir = move_dir.normalize();

  Rotate(eye, focus, angle, move_dir);
}

void RotateDown(Point4f &eye, Point4f &focus, float angle) {
  Vector4f v = camNode["upVector"].value().toVector4f();

  Vector4f w = focus - eye;  // Cam direction
  Vector4f move_dir;         // Cross(w,v)

  move_dir[0] = w[1] * v[2] - w[2] * v[1];
  move_dir[1] = w[2] * v[0] - w[0] * v[2];
  move_dir[2] = w[0] * v[1] - w[1] * v[0];
  move_dir[3] = 0.0;
  move_dir = move_dir.normalize();

  Rotate(eye, focus, -angle, move_dir);
}

/*
 * Temporary camera sync across mpi nodes until we have context consistent
 */
void SyncCamera(){

  unsigned char * v_new = new unsigned char[4 * sizeof(float)];

  camNode["eyePoint"].value().toPoint4f().pack(v_new);
  MPI_Bcast(v_new, 4, MPI_FLOAT, opengl_rank, MPI_COMM_WORLD);

  if(mpi_rank != opengl_rank)
    camNode["eyePoint"] = Point4f(v_new);

  camNode["focus"].value().toPoint4f().pack(v_new);
  MPI_Bcast(v_new, 4, MPI_FLOAT, opengl_rank, MPI_COMM_WORLD);

  if(mpi_rank != opengl_rank)
    camNode["focus"] = Point4f(v_new);

  camNode["upVector"].value().toVector4f().pack(v_new);
  MPI_Bcast(v_new, 4, MPI_FLOAT, opengl_rank, MPI_COMM_WORLD);

  if(mpi_rank != opengl_rank)
    camNode["upVector"] = Vector4f(v_new);

}

void UpdateCamera(Point4f focus, Point4f eye1, Vector4f up) {

  if (update){

      Point4f old_focus = camNode["focus"].value().toPoint4f();
      std::cout << "old_focus: " << old_focus[0] << " "
                << old_focus[1] << " "
                << old_focus[2] << endl;

      std::cout << "new_focus: " << focus[0] << " "
                << focus[1] << " "
                << focus[2] << endl;


      Point4f old_eyePoint = camNode["eyePoint"].value().toPoint4f();
      std::cout << "old_eyePoint: " << old_eyePoint[0] << " "
                << old_eyePoint[1] << " "
                << old_eyePoint[2] << endl;

      std::cout << "new_eyePoint: " << eye1[0] << " "
                << eye1[1] << " "
                << eye1[2] << endl;


      camNode["eyePoint"] = eye1;
      camNode["focus"] = focus;

        Vector4f v = (focus - eye1).normalize();
        float cross = v * up.normalize();
        if( cross == 1 || cross == -1) {
            if(up[1] == 1)
            up = Vector4f(0,0,1,0);
            else if(up[2]==1 || up[0] == 1)
            up = Vector4f(0,1,0,0);
        }

      camNode["upVector"] = up;

      unsigned char key = 'c';
      MPI_Bcast(&key, 1, MPI_UNSIGNED_CHAR, opengl_rank, MPI_COMM_WORLD);
      SyncCamera();

    }

  if (renderMode == BVH_RENDER_MODE) {
      Vector4f up = camNode["upVector"].value().toVector4f();
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      gluLookAt(eye1[0], eye1[1], eye1[2], focus[0], focus[1], focus[2], up[0],
          up[1], up[2]);
    }
}

static void mouseFunc(int button, int state, int x, int y) {
  if (button == 0) {
      if (state == GLUT_DOWN) {
          // Record start position
          mouseGrabLastX = x;
          mouseGrabLastY = y;
          mouseButton0 = 1;
        } else if (state == GLUT_UP) {
          mouseButton0 = 0;
        }
    } else if (button == 2) {
      if (state == GLUT_DOWN) {
          // Record start position
          mouseGrabLastX = x;
          mouseGrabLastY = y;
          mouseButton2 = 1;
        } else if (state == GLUT_UP) {
          mouseButton2 = 0;
        }
    }
}

static void motionFunc(int x, int y) {
  Point4f eye1 = camNode["eyePoint"].value().toPoint4f();
  Point4f focus = camNode["focus"].value().toPoint4f();
  Vector4f up = camNode["upVector"].value().toVector4f();

  const double minInterval = 0.2;
  if (mouseButton0) {
      // Check elapsed time since last update
      if (WallClockTime() - lastMouseUpdate > minInterval) {
          const int distX = x - mouseGrabLastX;
          const int distY = y - mouseGrabLastY;

          RotateDown(eye1, focus, 0.004f * distY * ROTATE_STEP);
          RotateRight(eye1, focus, 0.004f * distX * ROTATE_STEP);

          mouseGrabLastX = x;
          mouseGrabLastY = y;

          update = true;

          UpdateCamera(focus, eye1,up);

          glutPostRedisplay();
          lastMouseUpdate = WallClockTime();
        }
    }
}

void reshape(int w, int h) {
  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH rootNode = cntxt->getRootNode();
  gvt::core::DBNodeH filmNode = rootNode["Film"];

  width = w;
  height = h;


  //TODO: This also needs to be consistent across mpi nodes. Also this
  // used when instancing the tracer.
  filmNode["width"] = w;
  filmNode["height"] = h;


  update = true;

  glViewport(0, 0, (GLsizei)w, (GLsizei)h);

  if (renderMode == FILM_RENDER_MODE) {
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

    } else {
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluPerspective(45, (double)w / (double)h, .1, 100);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      Point4f eye1 = camNode["eyePoint"].value().toPoint4f();
      Point4f focus = camNode["focus"].value().toPoint4f();
      Vector4f up = camNode["upVector"].value().toVector4f();

      UpdateCamera( focus,  eye1,  up);

    }
}

void drawWireBox(gvt::render::data::primitives::Box3D &bbox) {
  float xmin = bbox.bounds[0][0];
  float ymin = bbox.bounds[0][1];
  float zmin = bbox.bounds[0][2];
  float xmax = bbox.bounds[1][0];
  float ymax = bbox.bounds[1][1];
  float zmax = bbox.bounds[1][2];

  glPushMatrix();
  glTranslatef(0.5f * (xmin + xmax), 0.5f * (ymin + ymax),
               0.5f * (zmin + zmax));
  glScalef(xmax - xmin, ymax - ymin, zmax - zmin);
  glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  glutWireCube(1.0f);
  glPopMatrix();
}

void RenderBVH() {
  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH rootNode = cntxt->getRootNode();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);

  for (gvt::core::DBNodeH instance : rootNode["Instances"].getChildren()) {
      Box3D *bbox = (Box3D *)instance["bbox"].value().toULongLong();
      drawWireBox(*bbox);
    }

  // glutSolidTeapot(.5);

  glFlush();
}

void Render() {
  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH rootNode = cntxt->getRootNode();
  gvt::core::DBNodeH camNode = rootNode["Camera"];
  gvt::core::DBNodeH filmNode = rootNode["Film"];

  // setup gvtCamera from database entries

  Point4f cameraposition = camNode["eyePoint"].value().toPoint4f();
  Point4f focus = camNode["focus"].value().toPoint4f();
  Vector4f up = camNode["upVector"].value().toVector4f();

  float fov = camNode["fov"].value().toFloat();
  mycamera.lookAt(cameraposition, focus, up);
  mycamera.setFOV(fov);
  mycamera.setFilmsize(filmNode["width"].value().toInteger(),
      filmNode["height"].value().toInteger());

  int schedType = rootNode["Schedule"]["type"].value().toInteger();

  mycamera.AllocateCameraRays();
  mycamera.generateRays();
  imageptr->clear();

  switch (schedType) {
    case gvt::render::scheduler::Image: {
        std::cout << "starting image scheduler" << std::endl;
        (*static_cast<gvt::render::algorithm::Tracer<ImageScheduler>*>(tracer))();
        break;
      }
    case gvt::render::scheduler::Domain: {
        std::cout << "starting domain scheduler" << std::endl;
        // *(gvt::render::algorithm::Tracer<DomainScheduler>* tracer)();
        (*static_cast<gvt::render::algorithm::Tracer<DomainScheduler>*>(tracer))();
        break;
      }
    default: {
        std::cout << "unknown schedule type provided: " << schedType << std::endl;
        break;
      }
    }

}

void RenderFilm() {
  unsigned char key = 'r';
  if (update) {
      MPI_Bcast(&key, 1, MPI_UNSIGNED_CHAR, opengl_rank, MPI_COMM_WORLD);
      Render();
      update = false;
    }
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glRasterPos2i(0, 0);
  glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, imagebuffer);
  glFlush();
}

void dispfunc(void) {
  switch (renderMode) {
    case BVH_RENDER_MODE: {
        RenderBVH();
        break;
      }
    case FILM_RENDER_MODE: {
        RenderFilm();
        break;
      }
    default: {
        cout << "unknown render mode" << endl;
        break;
      }
    }
}

void specialkey(int key, int x, int y) {
  Point4f eye1 = camNode["eyePoint"].value().toPoint4f();
  Point4f focus = camNode["focus"].value().toPoint4f();
  Vector4f up = camNode["upVector"].value().toVector4f();

  switch (key) {
    case GLUT_KEY_LEFT:  // translate camera left
      TranslateLeft(eye1, focus, MOVE_STEP);
      update = true;
      break;
    case GLUT_KEY_RIGHT:  // translate camera right
      TranslateRight(eye1, focus, MOVE_STEP);
      update = true;
      break;
    case GLUT_KEY_UP:  // translate camera right
      TranslateForward(eye1, focus, MOVE_STEP);
      update = true;
      break;
    case GLUT_KEY_DOWN:  // translate camera right
      TranslateBackward(eye1, focus, MOVE_STEP);
      update = true;
      break;
    default:
      break;
    }

  UpdateCamera(focus, eye1,up);

  glutPostRedisplay();
}

void UpdateRenderMode() {
  if (renderMode == BVH_RENDER_MODE) {
      renderMode = FILM_RENDER_MODE;

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

    } else {
      renderMode = BVH_RENDER_MODE;
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluPerspective(45, (double)width / (double)height, .1, 100);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      Point4f eye1 = camNode["eyePoint"].value().toPoint4f();
      Point4f focus = camNode["focus"].value().toPoint4f();
      Vector4f up = camNode["upVector"].value().toVector4f();

      UpdateCamera( focus,  eye1,  up);
    }
}

void keyboard(unsigned char key, int x, int y) {
  Point4f eye1 = camNode["eyePoint"].value().toPoint4f();
  Point4f focus = camNode["focus"].value().toPoint4f();
  Vector4f up = camNode["upVector"].value().toVector4f();

  switch (key) {
    case ESCAPE:
      MPI_Bcast(&key, 1, MPI_UNSIGNED_CHAR, opengl_rank, MPI_COMM_WORLD);
      if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();
      exit(0);

    case 'q':
      MPI_Bcast(&key, 1, MPI_UNSIGNED_CHAR, opengl_rank, MPI_COMM_WORLD);
      if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();
      exit(0);

    case 'w':
      RotateUp(eye1, focus, ROTATE_STEP);
      update = true;
      break;
    case 's':
      RotateDown(eye1, focus, ROTATE_STEP);
      update = true;
      break;
    case 'a':
      RotateLeft(eye1, focus, ROTATE_STEP);
      update = true;
      break;
    case 'd':
      RotateRight(eye1, focus, ROTATE_STEP);
      update = true;
      break;

    case 'm':
      UpdateRenderMode();
      break;
    default:
      // dont do anything
      break;
    }

  if (update) UpdateCamera(focus, eye1,up);

  glutPostRedisplay();
}

void ConfigSceneFromFile(string filename) {
  gvtapps::render::ConfigFileLoader cl(filename);
  gvt::render::data::Dataset &scene = cl.scene;

  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();

  gvt::core::DBNodeH root = cntxt->getRootNode();
  gvt::core::DBNodeH dataNodes = root["Data"];
  gvt::core::DBNodeH instNodes = root["Instances"];

  for (int i = 0; i < scene.domainSet.size(); i++) {
      Mesh *mesh = ((GeometryDomain *)scene.getDomain(i))->getMesh();

      gvt::core::DBNodeH meshNode =
          cntxt->createNodeFromType("Mesh", filename.c_str(), dataNodes.UUID());

      meshNode["file"] = filename;
      // mesh->computeBoundingBox();
      gvt::render::data::primitives::Box3D *bbox = mesh->getBoundingBox();
      meshNode["bbox"] = (unsigned long long)bbox;
      meshNode["ptr"] = (unsigned long long)mesh;

      // add instance
      gvt::core::DBNodeH instnode =
          cntxt->createNodeFromType("Instance", "inst", instNodes.UUID());
      Box3D *mbox = (Box3D *)meshNode["bbox"].value().toULongLong();
      instnode["id"] = i;
      instnode["meshRef"] = meshNode.UUID();
      auto m = new gvt::core::math::AffineTransformMatrix<float>(true);
      auto minv = new gvt::core::math::AffineTransformMatrix<float>(true);
      auto normi = new gvt::core::math::Matrix3f();
      instnode["mat"] = (unsigned long long)m;
      *minv = m->inverse();
      instnode["matInv"] = (unsigned long long)minv;
      *normi = m->upper33().inverse().transpose();
      instnode["normi"] = (unsigned long long)normi;
      auto il = (*m) * mbox->bounds[0];
      auto ih = (*m) * mbox->bounds[1];
      Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
      instnode["bbox"] = (unsigned long long)ibox;
      instnode["centroid"] = ibox->centroid();
    }

  // add lights, camera, and film to the database
  gvt::core::DBNodeH lightNode =
      cntxt->createNodeFromType("PointLight", "PointLight", root["Lights"].UUID());
  gvt::render::data::scene::PointLight *lp =
      (gvt::render::data::scene::PointLight *)scene.getLight(0);
  lightNode["position"] = Vector4f(lp->position);
  lightNode["color"] = Vector4f(lp->color);

  // camera
  gvt::core::DBNodeH _camNode = root["Camera"];
  _camNode["eyePoint"] = Point4f(scene.camera.getEye());
  _camNode["focus"] = Point4f(scene.camera.getFocus());
  _camNode["upVector"] = scene.camera.getUp();
  _camNode["fov"] = (float)(45.0 * M_PI / 180.0);  // TODO

  // film
  gvt::core::DBNodeH filmNode = root["Film"];

  filmNode["width"] = int(scene.camera.getFilmSizeWidth());
  filmNode["height"] = int(scene.camera.getFilmSizeHeight());
}

void ConfigSceneCubeCone() {
  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();

  gvt::core::DBNodeH root = cntxt->getRootNode();

  gvt::core::DBNodeH dataNodes = root["Data"];

  gvt::core::DBNodeH coneMeshNode =
      cntxt->createNodeFromType("Mesh", "conemesh", dataNodes.UUID());

  {
    Mesh *mesh = new Mesh(new Lambert(Vector4f(0.5, 0.5, 0.5, 1.0)));
    int numPoints = 7;
    Point4f points[7];
    points[0] = Point4f(0.5, 0.0, 0.0, 1.0);
    points[1] = Point4f(-0.5, 0.5, 0.0, 1.0);
    points[2] = Point4f(-0.5, 0.25, 0.433013, 1.0);
    points[3] = Point4f(-0.5, -0.25, 0.43013, 1.0);
    points[4] = Point4f(-0.5, -0.5, 0.0, 1.0);
    points[5] = Point4f(-0.5, -0.25, -0.433013, 1.0);
    points[6] = Point4f(-0.5, 0.25, -0.433013, 1.0);

    for (int i = 0; i < numPoints; i++) {
        mesh->addVertex(points[i]);
      }
    mesh->addFace(1, 2, 3);
    mesh->addFace(1, 3, 4);
    mesh->addFace(1, 4, 5);
    mesh->addFace(1, 5, 6);
    mesh->addFace(1, 6, 7);
    mesh->addFace(1, 7, 2);
    mesh->generateNormals();

    // calculate bbox
    Point4f lower = points[0], upper = points[0];
    for (int i = 1; i < numPoints; i++) {
        for (int j = 0; j < 3; j++) {
            lower[j] = (lower[j] < points[i][j]) ? lower[j] : points[i][j];
            upper[j] = (upper[j] > points[i][j]) ? upper[j] : points[i][j];
          }
      }
    Box3D *meshbbox = new gvt::render::data::primitives::Box3D(lower, upper);

    // add cone mesh to the database
    coneMeshNode["file"] = string("/fake/path/to/cone");
    coneMeshNode["bbox"] = (unsigned long long)meshbbox;
    coneMeshNode["ptr"] = (unsigned long long)mesh;
  }

  gvt::core::DBNodeH cubeMeshNode =
      cntxt->createNodeFromType("Mesh", "cubemesh", dataNodes.UUID());
  {
    Mesh *mesh = new Mesh(new Lambert(Vector4f(0.5, 0.5, 0.5, 1.0)));
    int numPoints = 8;
    Point4f points[8];
    points[0] = Point4f(-0.5, -0.5, 0.5, 1.0);
    points[1] = Point4f(0.5, -0.5, 0.5, 1.0);
    points[2] = Point4f(0.5, 0.5, 0.5, 1.0);
    points[3] = Point4f(-0.5, 0.5, 0.5, 1.0);
    points[4] = Point4f(-0.5, -0.5, -0.5, 1.0);
    points[5] = Point4f(0.5, -0.5, -0.5, 1.0);
    points[6] = Point4f(0.5, 0.5, -0.5, 1.0);
    points[7] = Point4f(-0.5, 0.5, -0.5, 1.0);

    for (int i = 0; i < numPoints; i++) {
        mesh->addVertex(points[i]);
      }
    // faces are 1 indexed
    mesh->addFace(1, 2, 3);
    mesh->addFace(1, 3, 4);
    mesh->addFace(2, 6, 7);
    mesh->addFace(2, 7, 3);
    mesh->addFace(6, 5, 8);
    mesh->addFace(6, 8, 7);
    mesh->addFace(5, 1, 4);
    mesh->addFace(5, 4, 8);
    mesh->addFace(1, 5, 6);
    mesh->addFace(1, 6, 2);
    mesh->addFace(4, 3, 7);
    mesh->addFace(4, 7, 8);
    mesh->generateNormals();

    // calculate bbox
    Point4f lower = points[0], upper = points[0];
    for (int i = 1; i < numPoints; i++) {
        for (int j = 0; j < 3; j++) {
            lower[j] = (lower[j] < points[i][j]) ? lower[j] : points[i][j];
            upper[j] = (upper[j] > points[i][j]) ? upper[j] : points[i][j];
          }
      }
    Box3D *meshbbox = new gvt::render::data::primitives::Box3D(lower, upper);

    // add cube mesh to the database
    cubeMeshNode["file"] = string("/fake/path/to/cube");
    cubeMeshNode["bbox"] = (unsigned long long)meshbbox;
    cubeMeshNode["ptr"] = (unsigned long long)mesh;
  }

  gvt::core::DBNodeH instNodes = root["Instances"];


  // create a NxM grid of alternating cones / cubes, offset using i and j
  int instId = 0;
  int ii[2] = {-2, 3};  // i range
  int jj[2] = {-2, 3};  // j range
  for (int i = ii[0]; i < ii[1]; i++) {
      for (int j = jj[0]; j < jj[1]; j++) {
          gvt::core::DBNodeH instnode =
              cntxt->createNodeFromType("Instance", "inst", instNodes.UUID());
          // gvt::core::DBNodeH meshNode = (instId % 2) ? coneMeshNode :
          // cubeMeshNode;
          gvt::core::DBNodeH meshNode = (instId % 2) ? cubeMeshNode : coneMeshNode;
          Box3D *mbox = (Box3D *)meshNode["bbox"].value().toULongLong();

          instnode["id"] = instId++;
          instnode["meshRef"] = meshNode.UUID();

          auto m = new gvt::core::math::AffineTransformMatrix<float>(true);
          auto minv = new gvt::core::math::AffineTransformMatrix<float>(true);
          auto normi = new gvt::core::math::Matrix3f();
          *m =
              *m * gvt::core::math::AffineTransformMatrix<float>::createTranslation(
                0.0, i * 0.5, j * 0.5);
          *m = *m * gvt::core::math::AffineTransformMatrix<float>::createScale(
                0.4, 0.4, 0.4);

          instnode["mat"] = (unsigned long long)m;
          *minv = m->inverse();
          instnode["matInv"] = (unsigned long long)minv;
          *normi = m->upper33().inverse().transpose();
          instnode["normi"] = (unsigned long long)normi;

          auto il = (*m) * mbox->bounds[0];
          auto ih = (*m) * mbox->bounds[1];
          Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
          instnode["bbox"] = (unsigned long long)ibox;
          instnode["centroid"] = ibox->centroid();
        }
    }

  // add lights, camera, and film to the database

  gvt::core::DBNodeH lightNode =
      cntxt->createNodeFromType("PointLight", "PointLight", root["Lights"].UUID());
  lightNode["position"] = Vector4f(1.0, 0.0, 0.0, 0.0);
  lightNode["color"] = Vector4f(1.0, 1.0, 1.0, 0.0);

  //  // second light just for fun
  //  gvt::core::DBNodeH lN2 = cntxt->createNodeFromType("PointLight",
  //                                                     "conelight",
  //                                                     lightNodes.UUID());
  //  lN2["position"] = Vector4f(2.0, 2.0, 2.0, 0.0);
  //  lN2["color"] = Vector4f(0.0, 0.0, 0.0, 0.0);

  gvt::core::DBNodeH _camNode = root["Camera"];

  _camNode["eyePoint"] = Point4f(4.0, 0.0, 0.0, 1.0);
  _camNode["focus"] = Point4f(0.0, 0.0, 0.0, 1.0);
  _camNode["upVector"] = Vector4f(0.0, 1.0, 0.0, 0.0);
  _camNode["fov"] = (float)(45.0 * M_PI / 180.0);

  gvt::core::DBNodeH filmNode = root["Film"];
  filmNode["width"] = 512;
  filmNode["height"] = 512;
}

void ConfigSceneCone() {
  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();

  gvt::core::DBNodeH root = cntxt->getRootNode();
  gvt::core::DBNodeH dataNodes = root["Data"];

  gvt::core::DBNodeH coneMeshNode =
      cntxt->createNodeFromType("Mesh", "conemesh", dataNodes.UUID());

  {
    Mesh *mesh = new Mesh(new Lambert(Vector4f(0.5, 0.5, 0.5, 1.0)));
    int numPoints = 7;
    Point4f points[7];
    points[0] = Point4f(0.5, 0.0, 0.0, 1.0);
    points[1] = Point4f(-0.5, 0.5, 0.0, 1.0);
    points[2] = Point4f(-0.5, 0.25, 0.433013, 1.0);
    points[3] = Point4f(-0.5, -0.25, 0.43013, 1.0);
    points[4] = Point4f(-0.5, -0.5, 0.0, 1.0);
    points[5] = Point4f(-0.5, -0.25, -0.433013, 1.0);
    points[6] = Point4f(-0.5, 0.25, -0.433013, 1.0);

    for (int i = 0; i < numPoints; i++) {
        mesh->addVertex(points[i]);
      }
    mesh->addFace(1, 2, 3);
    mesh->addFace(1, 3, 4);
    mesh->addFace(1, 4, 5);
    mesh->addFace(1, 5, 6);
    mesh->addFace(1, 6, 7);
    mesh->addFace(1, 7, 2);
    mesh->generateNormals();

    // calculate bbox
    Point4f lower = points[0], upper = points[0];
    for (int i = 1; i < numPoints; i++) {
        for (int j = 0; j < 3; j++) {
            lower[j] = (lower[j] < points[i][j]) ? lower[j] : points[i][j];
            upper[j] = (upper[j] > points[i][j]) ? upper[j] : points[i][j];
          }
      }
    Box3D *meshbbox = new gvt::render::data::primitives::Box3D(lower, upper);

    // add cone mesh to the database
    coneMeshNode["file"] = string("/fake/path/to/cone");
    coneMeshNode["bbox"] = (unsigned long long)meshbbox;
    coneMeshNode["ptr"] = (unsigned long long)mesh;
  }



  gvt::core::DBNodeH instnode =
      cntxt->createNodeFromType("Instance", "inst", root["Instances"].UUID());

  gvt::core::DBNodeH meshNode = coneMeshNode;

  Box3D *mbox = (Box3D *)meshNode["bbox"].value().toULongLong();

  instnode["id"] = 0;
  instnode["meshRef"] = meshNode.UUID();

  auto m = new gvt::core::math::AffineTransformMatrix<float>(true);
  auto minv = new gvt::core::math::AffineTransformMatrix<float>(true);
  auto normi = new gvt::core::math::Matrix3f();
  //          *m =
  //              *m *
  //              gvt::core::math::AffineTransformMatrix<float>::createTranslation(
  //                0.0, i * 0.5, j * 0.5);
  //          *m = *m *
  //          gvt::core::math::AffineTransformMatrix<float>::createScale(
  //                0.4, 0.4, 0.4);

  instnode["mat"] = (unsigned long long)m;
  *minv = m->inverse();
  instnode["matInv"] = (unsigned long long)minv;
  *normi = m->upper33().inverse().transpose();
  instnode["normi"] = (unsigned long long)normi;

  auto il = (*m) * mbox->bounds[0];
  auto ih = (*m) * mbox->bounds[1];
  Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
  instnode["bbox"] = (unsigned long long)ibox;
  instnode["centroid"] = ibox->centroid();

  // add lights, camera, and film to the database

  gvt::core::DBNodeH lightNode =
      cntxt->createNodeFromType("PointLight", "PointLight", root["Lights"].UUID());
  lightNode["position"] = Vector4f(1.0, 0.0, 0.0, 0.0);
  lightNode["color"] = Vector4f(1.0, 1.0, 1.0, 0.0);



  gvt::core::DBNodeH _camNode = root["Camera"];

  _camNode["eyePoint"] = Point4f(4.0, 0.0, 0.0, 1.0);
  _camNode["focus"] = Point4f(0.0, 0.0, 0.0, 1.0);
  _camNode["upVector"] = Vector4f(0.0, 1.0, 0.0, 0.0);
  _camNode["fov"] = (float)(45.0 * M_PI / 180.0);

  gvt::core::DBNodeH filmNode = root["Film"];

  filmNode["width"] = 512;
  filmNode["height"] = 512;
}

int main(int argc, char *argv[]) {
  unsigned char action;
  // mpi initialization

  mpi_rank = -1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Barrier(MPI_COMM_WORLD);
  opengl_rank = 0;

  string filename;

  if (argc > 1) {
      filename = argv[1];
    } else {
      cerr << " application requires input config file" << endl;
      if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();
      exit(1);
    }

  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();

  if (cntxt == NULL) {
      std::cout << "Something went wrong initializing the context" << std::endl;
      exit(0);
    }

  gvt::core::DBNodeH root = cntxt->getRootNode();

  cntxt->createNodeFromType("Data", "Data", root.UUID());
  cntxt->createNodeFromType("Instances", "Instances", root.UUID());
  cntxt->createNodeFromType("Lights", "Lights", root.UUID());
  cntxt->createNodeFromType("Camera", "Camera", root.UUID());
  cntxt->createNodeFromType("Film", "Film", root.UUID());


  ConfigSceneFromFile(filename);
  //ConfigSceneCubeCone();
  //ConfigSceneCone();


  gvt::core::DBNodeH schedNode =
      cntxt->createNodeFromType("Schedule", "Schedule", root.UUID());
  schedNode["type"] = gvt::render::scheduler::Image;
  // schedNode["type"] = gvt::render::scheduler::Domain;

#ifdef GVT_RENDER_ADAPTER_EMBREE
  int adapterType = gvt::render::adapter::Embree;
#elif GVT_RENDER_ADAPTER_MANTA
  int adapterType = gvt::render::adapter::Manta;
#elif GVT_RENDER_ADAPTER_OPTIX
  int adapterType = gvt::render::adapter::Optix;
#else
  GVT_DEBUG(DBG_ALWAYS, "ERROR: missing valid adapter");
#endif

  schedNode["adapter"] = adapterType;

  camNode = root["Camera"];
  width = root["Film"]["width"].value().toInteger();
  height = root["Film"]["height"].value().toInteger();

  imageptr = new Image(width, height, "spoot");
  imagebuffer = imageptr->GetBuffer();

  mycamera.AllocateCameraRays();


  int schedType = root["Schedule"]["type"].value().toInteger();
  switch (schedType) {
    case gvt::render::scheduler::Image:
      tracer = new gvt::render::algorithm::Tracer<ImageScheduler>(mycamera.rays,
                                                                  *imageptr);
      break;
    case gvt::render::scheduler::Domain:
      std::cout << "starting domain scheduler" << std::endl;
      tracer = new gvt::render::algorithm::Tracer<DomainScheduler>(
            mycamera.rays, *imageptr);
      break;
    default:
      std::cout << "unknown schedule type provided: " << schedType << std::endl;
      break;
    }

  Render();

  if (mpi_rank == opengl_rank) {  // max rank process does display

      glutInit(&argc, argv);

      glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH);
      glutInitWindowSize(width, height);
      glutInitWindowPosition(10, 10);
      glutCreateWindow(filename.c_str());

      glClearColor(0., 0., 0., 0.);

      glutDisplayFunc(dispfunc);
      glutSpecialFunc(specialkey);
      glutKeyboardFunc(keyboard);
      glutReshapeFunc(reshape);
      glutMouseFunc(mouseFunc);
      glutMotionFunc(motionFunc);

      /* set up depth-buffering */
      glEnable(GL_DEPTH_TEST);

      if (renderMode == FILM_RENDER_MODE) {
          glMatrixMode(GL_PROJECTION);
          glLoadIdentity();
          gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);
          glMatrixMode(GL_MODELVIEW);
          glLoadIdentity();

        } else {
          glMatrixMode(GL_PROJECTION);
          glLoadIdentity();
          gluPerspective(45, (double)width / (double)height, .1, 100);
          glMatrixMode(GL_MODELVIEW);
          glLoadIdentity();

          Point4f eye1 = camNode["eyePoint"].value().toPoint4f();
          Point4f focus = camNode["focus"].value().toPoint4f();
          Vector4f up = camNode["upVector"].value().toVector4f();

          update=true; //force up vector otho check, not applied to gtv first frame
          UpdateCamera( focus,  eye1,  up);
          update=false;
        }

      glutMainLoop();
    } else {  // loop and wait for further instructions
      while (1) {

          MPI_Bcast(&action, 1, MPI_CHAR, opengl_rank, MPI_COMM_WORLD);
          switch (action) {
            case ESCAPE:
              MPI_Finalize();
              break;
            case 'q':
              MPI_Finalize();
              break;
            case 'r':
              Render();
              break;
            case 'c':
              SyncCamera();
              break;
            default:
              break;
            }
        }
    }
  return 0;
}
