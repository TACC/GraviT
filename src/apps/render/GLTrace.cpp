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

#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

#include <tbb/task_scheduler_init.h>
#include <thread>

#include "ConfigFileLoader.h"
#include <gvt/core/Math.h>
#include <gvt/core/mpi/Wrapper.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/algorithm/Tracers.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/data/scene/gvtCamera.h>

#include "ParseCommandLine.h"

#if defined(__APPLE__)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <ply.h>

using namespace std;
using namespace gvtapps::render;
using namespace gvt::render::data::domain;
using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;
using namespace gvt::core::mpi;

using namespace gvt::render;
using namespace gvt::render::data::primitives;

typedef enum { BVH_RENDER_MODE, FILM_RENDER_MODE } render_mode;

#define ESCAPE 27
#define ROTATE_STEP .1
#define MOVE_STEP 0.1

// global variables used by glut callbacks.
//
GLubyte *imagebuffer;
static GLint width;
static GLint height;

int opengl_rank;
int mpi_rank;
bool update = false;
// render_mode renderMode = BVH_RENDER_MODE;
render_mode renderMode = FILM_RENDER_MODE;
bool cameraRotationMode = true; // true to rotate focus, false to rotate eye

gvt::core::DBNodeH camNode;

static int mouseButton0 = 0;
static int mouseButton2 = 0;
static int mouseGrabLastX = 0;
static int mouseGrabLastY = 0;
static double lastMouseUpdate = 0.0;
static bool printHelp = true;
gvt::render::algorithm::AbstractTrace *tracer;
gvtPerspectiveCamera mycamera;
Image *imageptr;
boost::timer::cpu_timer t_frame;
boost::timer::cpu_times lastFrameTime;

/*
 * Ply
 */
typedef struct Vertex {
  float x, y, z;
  float nx, ny, nz;
  void *other_props; /* other properties */
} Vertex;

typedef struct Face {
  unsigned char nverts; /* number of vertex indices in list */
  int *verts;           /* vertex index list */
  void *other_props;    /* other properties */
} Face;

PlyProperty vert_props[] = {
  /* list of property information for a vertex */
  { "x", Float32, Float32, offsetof(Vertex, x), 0, 0, 0, 0 },
  { "y", Float32, Float32, offsetof(Vertex, y), 0, 0, 0, 0 },
  { "z", Float32, Float32, offsetof(Vertex, z), 0, 0, 0, 0 },
  { "nx", Float32, Float32, offsetof(Vertex, nx), 0, 0, 0, 0 },
  { "ny", Float32, Float32, offsetof(Vertex, ny), 0, 0, 0, 0 },
  { "nz", Float32, Float32, offsetof(Vertex, nz), 0, 0, 0, 0 },
};

PlyProperty face_props[] = {
  /* list of property information for a face */
  { "vertex_indices", Int32, Int32, offsetof(Face, verts), 1, Uint8, Uint8, offsetof(Face, nverts) },
};

#define MIN(a, b) ((a < b) ? (a) : (b))
#define MAX(a, b) ((a > b) ? (a) : (b))

static Vertex **vlist;
static Face **flist;

inline double WallClockTime() {
#if defined(__linux__) || defined(__APPLE__) || defined(__CYGWIN__) || defined(__OpenBSD__) || defined(__FreeBSD__)
  struct timeval t;
  gettimeofday(&t, NULL);

  return t.tv_sec + t.tv_usec / 1000000.0;
#elif defined(WIN32)
  return GetTickCount() / 1000.0;
#else
#error "Unsupported Platform !!!"
#endif
}

void printString(void *font, const char *string) {
  int len, i;

  len = (int)strlen(string);
  for (i = 0; i < len; i++) glutBitmapCharacter(font, string[i]);
}

void PrintHelpString(const unsigned int x, const unsigned int y, const char *key, const char *msg) {
  glColor3f(0.9f, 0.9f, 0.5f);
  glRasterPos2i(x, y);
  printString(GLUT_BITMAP_8_BY_13, key);

  glColor3f(1.f, 1.f, 1.f);
  // To update raster color
  glRasterPos2i(x + glutBitmapLength(GLUT_BITMAP_8_BY_13, (unsigned char *)key), y);
  printString(GLUT_BITMAP_8_BY_13, ": ");
  printString(GLUT_BITMAP_8_BY_13, msg);
}

void PrintHelpAndSettings() {

  if (printHelp) {

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, width, 0, height);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(1.f, 1.f, 1.f);
    int fontOffset = 0;

    fontOffset += 15;
    PrintHelpString(15, fontOffset, "p", "write .ppm image");

    fontOffset += 15;
    PrintHelpString(15, fontOffset, "c", "switch camera rotation mode (rotate around eye or around focus)");

    fontOffset += 15;
    PrintHelpString(15, fontOffset, "a, s, d, w or mouse X/Y + mouse button 0", "rotate camera");

    fontOffset += 15;
    PrintHelpString(15, fontOffset, "arrow keys", "translate camera left, right, forward, and backward");

    fontOffset += 15;
    PrintHelpString(15, fontOffset, "r", "refresh frame");

    fontOffset += 15;
    PrintHelpString(15, fontOffset, "m", "switch between film and bvh render mode");

    fontOffset += 15;
    PrintHelpString(15, fontOffset, "h", "toggle Help");

    fontOffset += 15;
    PrintHelpString(15, fontOffset, "frame time", boost::timer::format(lastFrameTime).c_str());

    glPopMatrix();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glMatrixMode(GL_MODELVIEW);
  }
}

/*
 * TODO: Multiple static values re-calculated over and over in these
 * camera manipulation routines
 */

void Translate(glm::vec3 &eye, glm::vec3 &focus, const float k) {
  glm::vec3 v = camNode["upVector"].value().tovec3();

  glm::vec3 w = focus - eye; // Cam direction
  glm::vec3 move_dir;        // Cross(w,v)

  move_dir[0] = w[1] * v[2] - w[2] * v[1];
  move_dir[1] = w[2] * v[0] - w[0] * v[2];
  move_dir[2] = w[0] * v[1] - w[1] * v[0];
  move_dir = glm::normalize(move_dir);

  glm::vec3 t = k * move_dir;
  eye += t;
  focus += t;
}

void TranslateLeft(glm::vec3 &eye, glm::vec3 &focus, const float k) { Translate(eye, focus, -k); }

void TranslateRight(glm::vec3 &eye, glm::vec3 &focus, const float k) { Translate(eye, focus, k); }

void TranslateForward(glm::vec3 &eye, glm::vec3 &focus, const float k) {
  glm::vec3 t = k * (focus - eye);
  eye += t;
  focus += t;
}

void TranslateBackward(glm::vec3 &eye, glm::vec3 &focus, const float k) {
  glm::vec3 t = -k * (focus - eye);
  eye += t;
  focus += t;
}

void Rotate(glm::vec3 &eye, glm::vec3 &focus, const float angle, const glm::vec3 &axis) {
  glm::vec3 p = focus - eye;

  glm::vec3 t = angle * axis;

  // glm::mat4 mAA;
  // mAA = glm::mat4::createRotation(t[0], 1.0, 0.0, 0.0) * glm::mat4::createRotation(t[1], 0.0, 1.0, 0.0) *
  //       glm::mat4::createRotation(t[2], 0.0, 0.0, 1.0);

  glm::mat4 mAA = glm::rotate(glm::mat4(1.f), t[0], glm::vec3(1, 0, 0)) *
                  glm::rotate(glm::mat4(1.f), t[1], glm::vec3(0, 1, 0)) *
                  glm::rotate(glm::mat4(1.f), t[2], glm::vec3(0, 0, 1));

  // Rotate focus point

  if (cameraRotationMode)
    focus = eye + glm::vec3(mAA * glm::vec4(p, 0.f));
  else
    eye = focus + glm::vec3(mAA * glm::vec4(-p, 0.f));
}

void RotateLeft(glm::vec3 &eye, glm::vec3 &focus, float angle) {
  glm::vec3 v = camNode["upVector"].value().tovec3();

  glm::vec3 w = focus - eye; // Cam direction
  glm::vec3 move_dir_x;      // Cross(w,v)
  glm::vec3 move_dir_y;      // Cross(move_dir_x,w)

  move_dir_x[0] = w[1] * v[2] - w[2] * v[1];
  move_dir_x[1] = w[2] * v[0] - w[0] * v[2];
  move_dir_x[2] = w[0] * v[1] - w[1] * v[0];

  move_dir_x = glm::normalize(move_dir_x);

  move_dir_y[0] = move_dir_x[1] * w[2] - move_dir_x[2] * w[1];
  move_dir_y[1] = move_dir_x[2] * w[0] - move_dir_x[0] * w[2];
  move_dir_y[2] = move_dir_x[0] * w[1] - move_dir_x[1] * w[0];

  move_dir_y = glm::normalize(move_dir_y);

  Rotate(eye, focus, angle, move_dir_y);
}

void RotateRight(glm::vec3 &eye, glm::vec3 &focus, float angle) {
  glm::vec3 v = camNode["upVector"].value().tovec3();

  glm::vec3 w = focus - eye; // Cam direction
  glm::vec3 move_dir_x;      // Cross(w,v)
  glm::vec3 move_dir_y;      // Cross(move_dir_x,w)

  move_dir_x[0] = w[1] * v[2] - w[2] * v[1];
  move_dir_x[1] = w[2] * v[0] - w[0] * v[2];
  move_dir_x[2] = w[0] * v[1] - w[1] * v[0];

  move_dir_x = glm::normalize(move_dir_x);

  move_dir_y[0] = move_dir_x[1] * w[2] - move_dir_x[2] * w[1];
  move_dir_y[1] = move_dir_x[2] * w[0] - move_dir_x[0] * w[2];
  move_dir_y[2] = move_dir_x[0] * w[1] - move_dir_x[1] * w[0];

  move_dir_y = glm::normalize(move_dir_y);

  Rotate(eye, focus, -angle, move_dir_y);
}

void RotateUp(glm::vec3 &eye, glm::vec3 &focus, float angle) {
  glm::vec3 v = camNode["upVector"].value().tovec3();

  glm::vec3 w = focus - eye; // Cam direction
  glm::vec3 move_dir;        // Cross(w,v)

  move_dir[0] = w[1] * v[2] - w[2] * v[1];
  move_dir[1] = w[2] * v[0] - w[0] * v[2];
  move_dir[2] = w[0] * v[1] - w[1] * v[0];

  move_dir = glm::normalize(move_dir);

  Rotate(eye, focus, angle, move_dir);
}

void RotateDown(glm::vec3 &eye, glm::vec3 &focus, float angle) {
  glm::vec3 v = camNode["upVector"].value().tovec3();

  glm::vec3 w = focus - eye; // Cam direction
  glm::vec3 move_dir;        // Cross(w,v)

  move_dir[0] = w[1] * v[2] - w[2] * v[1];
  move_dir[1] = w[2] * v[0] - w[0] * v[2];
  move_dir[2] = w[0] * v[1] - w[1] * v[0];

  move_dir = glm::normalize(move_dir);

  Rotate(eye, focus, -angle, move_dir);
}

/*
 * Temporary camera film size across mpi nodes until we have context consistent
 */
void SyncFilmSize() {

  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH rootNode = cntxt->getRootNode();
  gvt::core::DBNodeH filmNode = rootNode["Film"];

  unsigned char *v_new = new unsigned char[4 * sizeof(float)];

  MPI_Bcast(&width, 1, MPI_INT, opengl_rank, MPI_COMM_WORLD);

  if (mpi_rank != opengl_rank) {
    filmNode["width"] = width;
  }

  MPI_Bcast(&height, 1, MPI_INT, opengl_rank, MPI_COMM_WORLD);

  if (mpi_rank != opengl_rank) {
    filmNode["height"] = height;
  }
}

/*
 * Temporary camera sync across mpi nodes until we have context consistent
 */
void SyncCamera() {

  unsigned char *v_new = new unsigned char[4 * sizeof(float)];

  glm::vec3 eye = camNode["eyePoint"].value().tovec3();
  MPI_Bcast(glm::value_ptr(eye), 4, MPI_FLOAT, opengl_rank, MPI_COMM_WORLD);
  if (mpi_rank != opengl_rank) camNode["eyePoint"] = eye;

  glm::vec3 focus = camNode["focus"].value().tovec3();
  MPI_Bcast(glm::value_ptr(focus), 4, MPI_FLOAT, opengl_rank, MPI_COMM_WORLD);
  if (mpi_rank != opengl_rank) camNode["focus"] = focus;

  glm::vec3 upVector = camNode["upVector"].value().tovec3();
  MPI_Bcast(glm::value_ptr(upVector), 4, MPI_FLOAT, opengl_rank, MPI_COMM_WORLD);
  if (mpi_rank != opengl_rank) camNode["upVector"] = upVector;
}

void UpdateCamera(glm::vec3 focus, glm::vec3 eye1, glm::vec3 up) {

  if (update) {

    //      glm::vec3 old_focus = camNode["focus"].value().tovec3();
    //      std::cout << "old_focus: " << old_focus[0] << " "
    //                << old_focus[1] << " "
    //                << old_focus[2] << endl;

    //      std::cout << "new_focus: " << focus[0] << " "
    //                << focus[1] << " "
    //                << focus[2] << endl;

    //      glm::vec3 old_eyePoint = camNode["eyePoint"].value().tovec3();
    //      std::cout << "old_eyePoint: " << old_eyePoint[0] << " "
    //                << old_eyePoint[1] << " "
    //                << old_eyePoint[2] << endl;

    //      std::cout << "new_eyePoint: " << eye1[0] << " "
    //                << eye1[1] << " "
    //                << eye1[2] << endl;

    camNode["eyePoint"] = eye1;
    camNode["focus"] = focus;

    // cam dir
    glm::vec3 v = glm::normalize(focus - eye1);
    float dot = glm::dot(v, glm::normalize(up));

    // if cam dir and up are converging to non-ortho,
    // update up vector to a correct ortho vector
    if (dot >= 0.6 || dot <= -0.6) {

      // Using vector orthgonal to cam dir, so calculating rotation axis
      // for cam dir
      glm::vec3 rot = glm::cross(v, glm::normalize(up));

      // rotate it 90 deg
      rot = (float)(90.0 * M_PI / 180.0) * rot;

      // ratation matrix
      // glm::mat4 mAA;
      // mAA = glm::mat4::createRotation(rot[0], 1.0, 0.0, 0.0) * glm::mat4::createRotation(rot[1], 0.0, 1.0, 0.0) *
      //       glm::mat4::createRotation(rot[2], 0.0, 0.0, 1.0);

      glm::mat4 mAA = glm::rotate(glm::mat4(1.f), rot[0], glm::vec3(1, 0, 0)) *
                      glm::rotate(glm::mat4(1.f), rot[1], glm::vec3(0, 1, 0)) *
                      glm::rotate(glm::mat4(1.f), rot[2], glm::vec3(0, 0, 1));

      // apply rotation
      up = glm::vec3(mAA * glm::vec4(v, 0.f));
    }

    camNode["upVector"] = up;

    unsigned char key = 'c';
    MPI_Bcast(&key, 1, MPI_UNSIGNED_CHAR, opengl_rank, MPI_COMM_WORLD);
    SyncCamera();
  }

  if (renderMode == BVH_RENDER_MODE) {
    glm::vec3 up = camNode["upVector"].value().tovec3();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    gluLookAt(eye1[0], eye1[1], eye1[2], focus[0], focus[1], focus[2], up[0], up[1], up[2]);
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
  glm::vec3 eye1 = camNode["eyePoint"].value().tovec3();
  glm::vec3 focus = camNode["focus"].value().tovec3();
  glm::vec3 up = camNode["upVector"].value().tovec3();

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

      UpdateCamera(focus, eye1, up);

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

  filmNode["width"] = w;
  filmNode["height"] = h;

  unsigned char key = 'f';
  MPI_Bcast(&key, 1, MPI_UNSIGNED_CHAR, opengl_rank, MPI_COMM_WORLD);
  SyncFilmSize();

  update = true;

  glViewport(0, 0, (GLsizei)w, (GLsizei)h);

  if (renderMode == FILM_RENDER_MODE) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

  } else {

    glm::vec3 eye1 = camNode["eyePoint"].value().tovec3();
    glm::vec3 focus = camNode["focus"].value().tovec3();
    glm::vec3 up = camNode["upVector"].value().tovec3();
    float fov = camNode["fov"].value().toFloat() * (180 / M_PI);
    // simply calculating the zfar according to camera view dir for now
    float zfar = (focus - eye1).length() * 1.5;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fov, (double)w / (double)h, .1, zfar);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    UpdateCamera(focus, eye1, up);
  }
}

void specialkey(int key, int x, int y) {
  glm::vec3 eye1 = camNode["eyePoint"].value().tovec3();
  glm::vec3 focus = camNode["focus"].value().tovec3();
  glm::vec3 up = camNode["upVector"].value().tovec3();

  switch (key) {
  case GLUT_KEY_LEFT:
    TranslateLeft(eye1, focus, MOVE_STEP);
    update = true;
    break;
  case GLUT_KEY_RIGHT:
    TranslateRight(eye1, focus, MOVE_STEP);
    update = true;
    break;
  case GLUT_KEY_UP:
    TranslateForward(eye1, focus, MOVE_STEP);
    update = true;
    break;
  case GLUT_KEY_DOWN:
    TranslateBackward(eye1, focus, MOVE_STEP);
    update = true;
    break;
  default:
    break;
  }

  UpdateCamera(focus, eye1, up);

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

    glm::vec3 eye1 = camNode["eyePoint"].value().tovec3();
    glm::vec3 focus = camNode["focus"].value().tovec3();
    glm::vec3 up = camNode["upVector"].value().tovec3();
    float fov = camNode["fov"].value().toFloat() * (180 / M_PI);
    // simply calculating the zfar according to camera view dir for now
    float zfar = (focus - eye1).length() * 1.5;

    renderMode = BVH_RENDER_MODE;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fov, (double)width / (double)height, .1, zfar);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    UpdateCamera(focus, eye1, up);
  }
}

void keyboard(unsigned char key, int x, int y) {
  glm::vec3 eye1 = camNode["eyePoint"].value().tovec3();
  glm::vec3 focus = camNode["focus"].value().tovec3();
  glm::vec3 up = camNode["upVector"].value().tovec3();

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

  case 'h':
    printHelp = (!printHelp);
    break;

  case 'p':
    imageptr->Write();
    break;

  case 'c':
    cameraRotationMode = (!cameraRotationMode);
    break;

  case 'r':
    update = true;
    break;

  default:
    // dont do anything
    break;
  }

  if (update) UpdateCamera(focus, eye1, up);

  glutPostRedisplay();
}

void drawWireBox(gvt::render::data::primitives::Box3D &bbox) {
  float xmin = bbox.bounds[0][0];
  float ymin = bbox.bounds[0][1];
  float zmin = bbox.bounds[0][2];
  float xmax = bbox.bounds[1][0];
  float ymax = bbox.bounds[1][1];
  float zmax = bbox.bounds[1][2];

  glPushMatrix();
  glTranslatef(0.5f * (xmin + xmax), 0.5f * (ymin + ymax), 0.5f * (zmin + zmax));
  glScalef(xmax - xmin, ymax - ymin, zmax - zmin);
  glutWireCube(1.0f);
  glPopMatrix();
}

void RenderBVH() {

  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH rootNode = cntxt->getRootNode();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);

  int schedType = rootNode["Schedule"]["type"].value().toInteger();

  std::map<int, size_t> mpiInstanceMap;
  if (schedType == gvt::render::scheduler::Domain)
    mpiInstanceMap = ((gvt::render::algorithm::Tracer<gvt::render::schedule::DomainScheduler> *)tracer)->mpiInstanceMap;

  size_t i = 0;
  for (gvt::core::DBNodeH instance :
       ((gvt::render::algorithm::Tracer<gvt::render::schedule::DomainScheduler> *)tracer)->instancenodes) {

    int mpiNode = 0;

    if (schedType == gvt::render::scheduler::Domain && MPI::COMM_WORLD.Get_size() > 1) {
      mpiNode = mpiInstanceMap[i++];
      GLfloat c[3] = { 0, 0, 0 };
      c[mpiNode % 3] = float((((mpiNode + 1) * 230) % 255)) / float(255);
      glColor3fv(c);

    } else
      glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

    Box3D *bbox = (Box3D *)instance["bbox"].value().toULongLong();
    drawWireBox(*bbox);
  }

  // glutSolidTeapot(.1);

  PrintHelpAndSettings();

  glutSwapBuffers();
}

void Render() {
  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH rootNode = cntxt->getRootNode();
  gvt::core::DBNodeH camNode = rootNode["Camera"];
  gvt::core::DBNodeH filmNode = rootNode["Film"];

  // setup gvtCamera from database entries

  glm::vec3 cameraposition = camNode["eyePoint"].value().tovec3();
  glm::vec3 focus = camNode["focus"].value().tovec3();
  glm::vec3 up = camNode["upVector"].value().tovec3();

  float fov = camNode["fov"].value().toFloat();
  mycamera.lookAt(cameraposition, focus, up);
  mycamera.setFOV(fov);
  mycamera.setFilmsize(filmNode["width"].value().toInteger(), filmNode["height"].value().toInteger());

  int schedType = rootNode["Schedule"]["type"].value().toInteger();

  mycamera.AllocateCameraRays();
  mycamera.generateRays();
  imageptr->clear();

  if (tracer->height != height || tracer->width != width) {

    imageptr = new Image(width, height, "spoot");
    imagebuffer = imageptr->GetBuffer();

    switch (schedType) {
    case gvt::render::scheduler::Image:
      tracer = new gvt::render::algorithm::Tracer<ImageScheduler>(mycamera.rays, *imageptr);
      break;
    case gvt::render::scheduler::Domain:
      std::cout << "starting domain scheduler" << std::endl;
      tracer = new gvt::render::algorithm::Tracer<DomainScheduler>(mycamera.rays, *imageptr);
      break;
    default:
      std::cout << "unknown schedule type provided: " << schedType << std::endl;
      break;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  t_frame.start();
  switch (schedType) {
  case gvt::render::scheduler::Image: {
    std::cout << "starting image scheduler" << std::endl;
    (*static_cast<gvt::render::algorithm::Tracer<ImageScheduler> *>(tracer))();
    break;
  }
  case gvt::render::scheduler::Domain: {
    std::cout << "starting domain scheduler" << std::endl;
    // *(gvt::render::algorithm::Tracer<DomainScheduler>* tracer)();
    (*static_cast<gvt::render::algorithm::Tracer<DomainScheduler> *>(tracer))();
    break;
  }
  default: {
    std::cout << "unknown schedule type provided: " << schedType << std::endl;
    break;
  }
  }

  t_frame.stop();
  lastFrameTime = t_frame.elapsed();
}

void RenderFilm() {
  unsigned char key = 'r';
  if (update) {
    MPI_Bcast(&key, 1, MPI_UNSIGNED_CHAR, opengl_rank, MPI_COMM_WORLD);
    Render();
    update = false;
  }
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  PrintHelpAndSettings();

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  glRasterPos2i(0, 0);
  glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, imagebuffer);

  glutSwapBuffers();
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

void ConfigSceneFromFile(std::string filename) {
// NEed to modify COnfigLo
#if 0
  gvtapps::render::ConfigFileLoader cl(filename);
  gvt::render::data::Dataset &scene = cl.scene;

  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();

  gvt::core::DBNodeH root = cntxt->getRootNode();
  gvt::core::DBNodeH dataNodes = root["Data"];
  gvt::core::DBNodeH instNodes = root["Instances"];

  for (int i = 0; i < scene.domainSet.size(); i++) {
    Mesh *mesh = ((GeometryDomain *)scene.getDomain(i))->getMesh();

    gvt::core::DBNodeH meshNode = cntxt->createNodeFromType("Mesh", filename.c_str(), dataNodes.UUID());

    meshNode["file"] = filename;
    // mesh->computeBoundingBox();
    gvt::render::data::primitives::Box3D *bbox = mesh->getBoundingBox();
    meshNode["bbox"] = (unsigned long long)bbox;
    meshNode["ptr"] = (unsigned long long)mesh;

    // add instance
    gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance", "inst", instNodes.UUID());
    Box3D *mbox = (Box3D *)meshNode["bbox"].value().toULongLong();
    instnode["id"] = i;
    instnode["meshRef"] = meshNode.UUID();
    auto m = new glm::mat4(true);
    auto minv = new glm::mat4(true);
    auto normi = new glm::mat3();
    instnode["mat"] = (unsigned long long)m;
    *minv = glm::inverse(*m);
    instnode["matInv"] = (unsigned long long)minv;
    *normi = glm::transpose(glm::inverse(glm::mat3(*m)));
    instnode["normi"] = (unsigned long long)normi;
    auto il = (*m) * mbox->bounds[0];
    auto ih = (*m) * mbox->bounds[1];
    Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
    instnode["bbox"] = (unsigned long long)ibox;
    instnode["centroid"] = ibox->centroid();
  }

  // add lights, camera, and film to the database
  gvt::core::DBNodeH lightNode = cntxt->createNodeFromType("PointLight", "PointLight", root["Lights"].UUID());
  gvt::render::data::scene::PointLight *lp = (gvt::render::data::scene::PointLight *)scene.getLight(0);
  lightNode["position"] = glm::vec3(lp->position);
  lightNode["color"] = glm::vec3(lp->color);

  // camera
  gvt::core::DBNodeH _camNode = root["Camera"];
  _camNode["eyePoint"] = glm::vec3(scene.camera.getEye());
  _camNode["focus"] = glm::vec3(scene.camera.getFocus());
  _camNode["upVector"] = scene.camera.getUp();
  _camNode["fov"] = (float)(45.0 * M_PI / 180.0); // TODO

  // film
  gvt::core::DBNodeH filmNode = root["Film"];

  filmNode["width"] = int(scene.camera.getFilmSizeWidth());
  filmNode["height"] = int(scene.camera.getFilmSizeHeight());
#endif
}

void ConfigSceneCubeCone() {
  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();

  gvt::core::DBNodeH root = cntxt->getRootNode();

  gvt::core::DBNodeH dataNodes = root["Data"];

  gvt::core::DBNodeH coneMeshNode = cntxt->createNodeFromType("Mesh", "conemesh", dataNodes.UUID());

  {
    Mesh *mesh = new Mesh(new Lambert(glm::vec3(0.5, 0.5, 0.5)));
    int numPoints = 7;
    glm::vec3 points[7];
    points[0] = glm::vec3(0.5, 0.0, 0.0);
    points[1] = glm::vec3(-0.5, 0.5, 0.0);
    points[2] = glm::vec3(-0.5, 0.25, 0.433013);
    points[3] = glm::vec3(-0.5, -0.25, 0.43013);
    points[4] = glm::vec3(-0.5, -0.5, 0.0);
    points[5] = glm::vec3(-0.5, -0.25, -0.433013);
    points[6] = glm::vec3(-0.5, 0.25, -0.433013);

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
    glm::vec3 lower = points[0], upper = points[0];
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

  gvt::core::DBNodeH cubeMeshNode = cntxt->createNodeFromType("Mesh", "cubemesh", dataNodes.UUID());
  {
    Mesh *mesh = new Mesh(new Lambert(glm::vec3(0.5, 0.5, 0.5)));
    int numPoints = 8;
    glm::vec3 points[8];
    points[0] = glm::vec3(-0.5, -0.5, 0.5);
    points[1] = glm::vec3(0.5, -0.5, 0.5);
    points[2] = glm::vec3(0.5, 0.5, 0.5);
    points[3] = glm::vec3(-0.5, 0.5, 0.5);
    points[4] = glm::vec3(-0.5, -0.5, -0.5);
    points[5] = glm::vec3(0.5, -0.5, -0.5);
    points[6] = glm::vec3(0.5, 0.5, -0.5);
    points[7] = glm::vec3(-0.5, 0.5, -0.5);

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
    glm::vec3 lower = points[0], upper = points[0];
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
  int ii[2] = { -2, 3 }; // i range
  int jj[2] = { -2, 3 }; // j range
  for (int i = ii[0]; i < ii[1]; i++) {
    for (int j = jj[0]; j < jj[1]; j++) {
      gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance", "inst", instNodes.UUID());
      // gvt::core::DBNodeH meshNode = (instId % 2) ? coneMeshNode :
      // cubeMeshNode;
      gvt::core::DBNodeH meshNode = (instId % 2) ? cubeMeshNode : coneMeshNode;
      Box3D *mbox = (Box3D *)meshNode["bbox"].value().toULongLong();

      instnode["id"] = instId++;
      instnode["meshRef"] = meshNode.UUID();

      auto m = new glm::mat4(1);
      auto minv = new glm::mat4(1);
      auto normi = new glm::mat3(1);
      *m = glm::translate(*m, glm::vec3(0.0, i * 0.5, j * 0.5));
      *m = glm::scale(*m, glm::vec3(0.4, 0.4, 0.4));

      instnode["mat"] = (unsigned long long)m;
      *minv = glm::inverse(*m);
      instnode["matInv"] = (unsigned long long)minv;
      *normi = glm::transpose(glm::inverse(glm::mat3(*m)));
      instnode["normi"] = (unsigned long long)normi;

      auto il = glm::vec3((*m) * glm::vec4(mbox->bounds[0], 1.f));
      auto ih = glm::vec3((*m) * glm::vec4(mbox->bounds[1], 1.f));
      Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
      instnode["bbox"] = (unsigned long long)ibox;
      instnode["centroid"] = ibox->centroid();
    }
  }

  // add lights, camera, and film to the database

  gvt::core::DBNodeH lightNode = cntxt->createNodeFromType("PointLight", "PointLight", root["Lights"].UUID());
  lightNode["position"] = glm::vec3(1.0, 0.0, 0.0);
  lightNode["color"] = glm::vec3(1.0, 1.0, 1.0);

  gvt::core::DBNodeH _camNode = root["Camera"];

  _camNode["eyePoint"] = glm::vec3(4.0, 0.0, 0.0);
  _camNode["focus"] = glm::vec3(0.0, 0.0, 0.0);
  _camNode["upVector"] = glm::vec3(0.0, 1.0, 0.0);
  _camNode["fov"] = (float)(45.0 * M_PI / 180.0);

  gvt::core::DBNodeH filmNode = root["Film"];
  filmNode["width"] = 512;
  filmNode["height"] = 512;
}

void ConfigSceneCone() {
  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();

  gvt::core::DBNodeH root = cntxt->getRootNode();
  gvt::core::DBNodeH dataNodes = root["Data"];

  gvt::core::DBNodeH coneMeshNode = cntxt->createNodeFromType("Mesh", "conemesh", dataNodes.UUID());

  {
    Mesh *mesh = new Mesh(new Lambert(glm::vec3(0.5, 0.5, 0.5)));
    int numPoints = 7;
    glm::vec3 points[7];
    points[0] = glm::vec3(0.5, 0.0, 0.0);
    points[1] = glm::vec3(-0.5, 0.5, 0.0);
    points[2] = glm::vec3(-0.5, 0.25, 0.433013);
    points[3] = glm::vec3(-0.5, -0.25, 0.43013);
    points[4] = glm::vec3(-0.5, -0.5, 0.0);
    points[5] = glm::vec3(-0.5, -0.25, -0.433013);
    points[6] = glm::vec3(-0.5, 0.25, -0.433013);

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
    glm::vec3 lower = points[0], upper = points[0];
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

  gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance", "inst", root["Instances"].UUID());

  gvt::core::DBNodeH meshNode = coneMeshNode;

  Box3D *mbox = (Box3D *)meshNode["bbox"].value().toULongLong();

  instnode["id"] = 0;
  instnode["meshRef"] = meshNode.UUID();

  auto m = new glm::mat4(true);
  auto minv = new glm::mat4(true);
  auto normi = new glm::mat3();
  //          *m =
  //              *m *
  //              glm::mat4::createTranslation(
  //                0.0, i * 0.5, j * 0.5);
  //          *m = *m *
  //          glm::mat4::createScale(
  //                0.4, 0.4, 0.4);

  instnode["mat"] = (unsigned long long)m;
  *minv = glm::inverse(*m);
  instnode["matInv"] = (unsigned long long)minv;
  *normi = glm::transpose(glm::inverse(glm::mat3(*m)));
  instnode["normi"] = (unsigned long long)normi;

  auto il = glm::vec3((*m) * glm::vec4(mbox->bounds[0], 1.f));
  auto ih = glm::vec3((*m) * glm::vec4(mbox->bounds[1], 1.f));
  Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
  instnode["bbox"] = (unsigned long long)ibox;
  instnode["centroid"] = ibox->centroid();

  // add lights, camera, and film to the database

  gvt::core::DBNodeH lightNode = cntxt->createNodeFromType("PointLight", "PointLight", root["Lights"].UUID());
  lightNode["position"] = glm::vec3(1.0, 0.0, 0.0);
  lightNode["color"] = glm::vec3(1.0, 1.0, 1.0);

  gvt::core::DBNodeH _camNode = root["Camera"];

  _camNode["eyePoint"] = glm::vec3(4.0, 0.0, 0.0);
  _camNode["focus"] = glm::vec3(0.0, 0.0, 0.0);
  _camNode["upVector"] = glm::vec3(0.0, 1.0, 0.0);
  _camNode["fov"] = (float)(45.0 * M_PI / 180.0);

  gvt::core::DBNodeH filmNode = root["Film"];

  filmNode["width"] = 512;
  filmNode["height"] = 512;
}

void ConfigEnzo(std::string rootdir) {

  // mess I use to open and read the ply file with the c utils I found.
  PlyFile *in_ply;
  Vertex *vert;
  Face *face;
  int elem_count, nfaces, nverts;
  int i, j, k;
  float xmin, ymin, zmin, xmax, ymax, zmax;
  char *elem_name;
  ;
  FILE *myfile;
  char txt[16];
  std::string temp;
  std::string filename, filepath;

  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();

  gvt::core::DBNodeH root = cntxt->getRootNode();
  gvt::core::DBNodeH dataNodes = root["Data"];

  // Enzo isosurface...
  for (k = 0; k < 8; k++) {
    sprintf(txt, "%d", k);
    filename = "block";
    filename += txt;
    gvt::core::DBNodeH EnzoMeshNode = cntxt->createNodeFromType("Mesh", filename.c_str(), dataNodes.UUID());
    // read in some ply data and get ready to load it into the mesh
    // filepath = rootdir + "block" + std::string(txt) + ".ply";
    filepath = rootdir + filename + ".ply";
    myfile = fopen(filepath.c_str(), "r");
    in_ply = read_ply(myfile);
    for (i = 0; i < in_ply->num_elem_types; i++) {
      elem_name = setup_element_read_ply(in_ply, i, &elem_count);
      temp = elem_name;
      if (temp == "vertex") {
        vlist = (Vertex **)malloc(sizeof(Vertex *) * elem_count);
        nverts = elem_count;
        setup_property_ply(in_ply, &vert_props[0]);
        setup_property_ply(in_ply, &vert_props[1]);
        setup_property_ply(in_ply, &vert_props[2]);
        for (j = 0; j < elem_count; j++) {
          vlist[j] = (Vertex *)malloc(sizeof(Vertex));
          get_element_ply(in_ply, (void *)vlist[j]);
        }
      } else if (temp == "face") {
        flist = (Face **)malloc(sizeof(Face *) * elem_count);
        nfaces = elem_count;
        setup_property_ply(in_ply, &face_props[0]);
        for (j = 0; j < elem_count; j++) {
          flist[j] = (Face *)malloc(sizeof(Face));
          get_element_ply(in_ply, (void *)flist[j]);
        }
      }
    }
    close_ply(in_ply);
    // smoosh data into the mesh object
    {
      Mesh *mesh = new Mesh(new Lambert(glm::vec3(1.0, 1.0, 1.0)));
      vert = vlist[0];
      xmin = vert->x;
      ymin = vert->y;
      zmin = vert->z;
      xmax = vert->x;
      ymax = vert->y;
      zmax = vert->z;

      for (i = 0; i < nverts; i++) {
        vert = vlist[i];
        xmin = MIN(vert->x, xmin);
        ymin = MIN(vert->y, ymin);
        zmin = MIN(vert->z, zmin);
        xmax = MAX(vert->x, xmax);
        ymax = MAX(vert->y, ymax);
        zmax = MAX(vert->z, zmax);
        mesh->addVertex(glm::vec3(vert->x, vert->y, vert->z));
      }
      glm::vec3 lower(xmin, ymin, zmin);
      glm::vec3 upper(xmax, ymax, zmax);
      Box3D *meshbbox = new gvt::render::data::primitives::Box3D(lower, upper);
      // add faces to mesh
      for (i = 0; i < nfaces; i++) {
        face = flist[i];
        mesh->addFace(face->verts[0] + 1, face->verts[1] + 1, face->verts[2] + 1);
      }
      mesh->generateNormals();
      // add Enzo mesh to the database
      // EnzoMeshNode["file"] = string("/work/01197/semeraro/maverick/DAVEDATA/EnzoPlyDATA/Block0.ply");
      EnzoMeshNode["file"] = string(filepath);
      EnzoMeshNode["bbox"] = (unsigned long long)meshbbox;
      EnzoMeshNode["ptr"] = (unsigned long long)mesh;
    }
    // add instance
    gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance", "inst", root["Instances"].UUID());
    gvt::core::DBNodeH meshNode = EnzoMeshNode;
    Box3D *mbox = (Box3D *)meshNode["bbox"].value().toULongLong();
    instnode["id"] = k;
    instnode["meshRef"] = meshNode.UUID();
    auto m = new glm::mat4(1.f);
    auto minv = new glm::mat4(1.f);
    auto normi = new glm::mat3(1.f);
    instnode["mat"] = (unsigned long long)m;
    *minv = glm::inverse(*m);
    instnode["matInv"] = (unsigned long long)minv;
    *normi = glm::transpose(glm::inverse(glm::mat3(*m)));
    instnode["normi"] = (unsigned long long)normi;
    auto il = glm::vec3((*m) * glm::vec4(mbox->bounds[0], 1.f));
    auto ih = glm::vec3((*m) * glm::vec4(mbox->bounds[1], 1.f));
    Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
    instnode["bbox"] = (unsigned long long)ibox;
    instnode["centroid"] = ibox->centroid();
  }

  // add lights, camera, and film to the database
  gvt::core::DBNodeH lightNodes = root["Lights"];
  gvt::core::DBNodeH lightNode = cntxt->createNodeFromType("PointLight", "conelight", lightNodes.UUID());
  lightNode["position"] = glm::vec3(512.0, 512.0, 2048.0);
  lightNode["color"] = glm::vec3(1.0, 1.0, 1.0);
  // camera
  gvt::core::DBNodeH camNode = root["Camera"];
  camNode["eyePoint"] = glm::vec3(512.0, 512.0, 4096.0);
  camNode["focus"] = glm::vec3(512.0, 512.0, 0.0);
  camNode["upVector"] = glm::vec3(0.0, 1.0, 0.0);
  camNode["fov"] = (float)(25.0 * M_PI / 180.0);
  // film
  gvt::core::DBNodeH filmNode = root["Film"];
  filmNode["width"] = 1900;
  filmNode["height"] = 1080;
}

int main(int argc, char *argv[]) {
  tbb::task_scheduler_init init(std::thread::hardware_concurrency());
  unsigned char action;
  // mpi initialization

  ParseCommandLine cmd("glTrace");

  cmd.addoption("wsize", ParseCommandLine::INT, "Window size", 2);
  cmd.addoption("eye", ParseCommandLine::FLOAT, "Camera position", 3);
  cmd.addoption("look", ParseCommandLine::FLOAT, "Camera look at", 3);
  cmd.addoption("enzo", ParseCommandLine::PATH, "Data path");
  cmd.addoption("simple", ParseCommandLine::NONE, "Use embeded scene", 0);
  cmd.addoption("scene", ParseCommandLine::PATH, "Use scene file");
  cmd.addoption("image", ParseCommandLine::NONE, "Use embeded scene", 0);
  cmd.addoption("domain", ParseCommandLine::NONE, "Use embeded scene", 0);

  cmd.addconflict("enzo", "simple");
  cmd.addconflict("enzo", "scene");
  cmd.addconflict("simple", "scene");
  cmd.addconflict("image", "domain");

  // cmd.addrequire("simple", "eye");
  // cmd.addrequire("simple", "look");

  cmd.parse(argc, argv);

  mpi_rank = -1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Barrier(MPI_COMM_WORLD);
  opengl_rank = 0;

  // string filename;
  //
  // if (argc > 1) {
  //   filename = argv[1];
  // } else {
  //   cerr << " application requires input config file" << endl;
  //   if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();
  //   exit(1);
  // }

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

  if (cmd.isSet("enzo"))
    ConfigEnzo(cmd.get<std::string>("enzo"));
  else if (cmd.isSet("scene"))
    ConfigSceneFromFile(cmd.get<std::string>("scene"));
  else
    ConfigSceneCubeCone();
  //
  // ConfigSceneCubeCone();
  // ConfigSceneCone();
  // ConfigEnzo();

  gvt::core::DBNodeH schedNode = cntxt->createNodeFromType("Schedule", "Schedule", root.UUID());
  if (cmd.isSet("domain"))
    schedNode["type"] = gvt::render::scheduler::Domain;
  else
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

  if (cmd.isSet("wsize")) {
    std::vector<int> wsize = cmd.getValue<int>("wsize");
    root["Film"]["width"] = wsize[0];
    root["Film"]["height"] = wsize[1];
  }

  if (cmd.isSet("eye")) {
    std::vector<float> eye = cmd.getValue<float>("eye");
    camNode["eyePoint"] = glm::vec3(eye[0], eye[1], eye[2]);
  }

  if (cmd.isSet("look")) {
    std::vector<float> eye = cmd.getValue<float>("look");
    camNode["focus"] = glm::vec3(eye[0], eye[1], eye[2]);
  }

  width = root["Film"]["width"].value().toInteger();
  height = root["Film"]["height"].value().toInteger();

  imageptr = new Image(width, height, "spoot");
  imagebuffer = imageptr->GetBuffer();

  mycamera.setFilmsize(root["Film"]["width"].value().toInteger(), root["Film"]["height"].value().toInteger());

  mycamera.AllocateCameraRays();

  int schedType = root["Schedule"]["type"].value().toInteger();
  switch (schedType) {
  case gvt::render::scheduler::Image:
    tracer = new gvt::render::algorithm::Tracer<ImageScheduler>(mycamera.rays, *imageptr);
    break;
  case gvt::render::scheduler::Domain:
    std::cout << "starting domain scheduler" << std::endl;
    tracer = new gvt::render::algorithm::Tracer<DomainScheduler>(mycamera.rays, *imageptr);
    break;
  default:
    std::cout << "unknown schedule type provided: " << schedType << std::endl;
    break;
  }

  // Render();

  if (mpi_rank == opengl_rank) { // max rank process does display

    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(width, height);
    glutInitWindowPosition(10, 10);
    glutCreateWindow("GL Render");

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

      glm::vec3 eye1 = camNode["eyePoint"].value().tovec3();
      glm::vec3 focus = camNode["focus"].value().tovec3();
      glm::vec3 up = camNode["upVector"].value().tovec3();
      float fov = camNode["fov"].value().toFloat() * (180 / M_PI);
      // simply calculating the zfar according to camera view dir for now
      float zfar = (focus - eye1).length() * 1.5;

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluPerspective(fov, (double)width / (double)height, .1, zfar);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      update = true; // force up vector otho check, not applied to gtv first frame
      UpdateCamera(focus, eye1, up);
      update = false;
    }

    glutMainLoop();
  } else { // loop and wait for further instructions
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
      case 'f':
        SyncFilmSize();
        break;
      default:
        break;
      }
    }
  }
  return 0;
}
