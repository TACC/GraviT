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


typedef enum {BVH_RENDER_MODE, FILM_RENDER_MODE} render_mode;

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
Image *imageptr;
int master;
bool update = false;
render_mode renderMode = BVH_RENDER_MODE;
gvt::core::DBNodeH camNode;

static int mouseButton0 = 0;
static int mouseButton2 = 0;
static int mouseGrabLastX = 0;
static int mouseGrabLastY = 0;
static double lastMouseUpdate = 0.0;

inline double WallClockTime() {
#if defined(__linux__) || defined(__APPLE__) || defined(__CYGWIN__) || defined(__OpenBSD__) || defined(__FreeBSD__)
  struct timeval t;
  gettimeofday(&t, NULL);

  return t.tv_sec + t.tv_usec / 1000000.0;
#elif defined (WIN32)
  return GetTickCount() / 1000.0;
#else
#error "Unsupported Platform !!!"
#endif
}

void Translate(Point4f& eye, Point4f& focus,const Point4f &t) {
  eye += t;
  focus += t;
}

void  TranslateLeft(Point4f& eye, Point4f& focus, const float k) {
  Point4f t = Point4f(-k, 0, 0,0);
  Translate(eye,focus, t);
}

void  TranslateRight(Point4f& eye, Point4f& focus, const float k) {
  Point4f t = Point4f(k, 0, 0,0);
  Translate(eye,focus, t);
}

void  TranslateForward(Point4f& eye, Point4f& focus, const float k) {
  Point4f t = k *(focus - eye);
  Translate(eye,focus, t);
}

void  TranslateBackward(Point4f& eye, Point4f& focus, const float k) {
  Point4f t = -k *(focus - eye);
  Translate(eye,focus, t);
}

void Rotate(Point4f& eye, Point4f& focus,
            const float angle, const Point4f& axis) {

  Vector4f p = focus - eye;

  gvt::core::math::Vector4f t = angle * axis;

  gvt::core::math::AffineTransformMatrix<float> mAA;
  mAA = gvt::core::math::AffineTransformMatrix<float>::createRotation(
        t[0], 1.0, 0.0, 0.0) *
      gvt::core::math::AffineTransformMatrix<float>::createRotation(
        t[1], 0.0, 1.0, 0.0) *
      gvt::core::math::AffineTransformMatrix<float>::createRotation(
        t[2], 0.0,  0.0, 1.0);

  gvt::core::math::AffineTransformMatrix<float> id;

  // Rotate focus point
  focus = eye + mAA*p;
}

void RotateLeft(Point4f& eye, Point4f& focus, float angle) {
  Rotate( eye, focus,angle, Point4f(0, 1, 0,0));
}

void RotateRight(Point4f& eye, Point4f& focus, float angle) {
  Rotate( eye, focus,-angle, Point4f(0, 1, 0,0));
}

void RotateUp(Point4f& eye, Point4f& focus, float angle) {
  Rotate( eye, focus, angle, Point4f(1, 0, 0,0));
}

void RotateDown(Point4f& eye, Point4f& focus, float angle) {
  Rotate( eye, focus, -angle, Point4f(1, 0, 0,0));
}

void UpdateCamera(Point4f focus, Point4f eye1)
{
  Point4f eye_o = gvt::core::variant_toPoint4f(camNode["eyePoint"].value());
  Point4f focus_o = gvt::core::variant_toPoint4f(camNode["focus"].value());

  cout << "old eye : " << eye_o[0] << " "<< eye_o[1] << " "<< eye_o[2] << endl;
  cout << "new eye : " << eye1[0] << " "<< eye1[1] << " "<< eye1[2] << endl;
  cout << "old focus : " << focus_o[0] << " "<< focus_o[1] << " "<< focus_o[2] << endl;
  cout << "new focus : " << focus[0] << " "<< focus[1] << " "<< focus[2] << endl;

  camNode["eyePoint"] = eye1;
  camNode["focus"] = focus;

  if (renderMode == BVH_RENDER_MODE){

      Point4f up = gvt::core::variant_toVector4f(camNode["upVector"].value());
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      gluLookAt(eye1[0],eye1[1],eye1[2],
          focus[0],focus[1],focus[2],
          up[0],up[1],up[2]);
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

  Point4f eye1 = gvt::core::variant_toPoint4f(camNode["eyePoint"].value());
  Point4f focus = gvt::core::variant_toPoint4f(camNode["focus"].value());

  const double minInterval = 0.2;
  if (mouseButton0) {
      // Check elapsed time since last update
      if (WallClockTime() - lastMouseUpdate > minInterval) {
          const int distX = x - mouseGrabLastX;
          const int distY = y - mouseGrabLastY;

          RotateDown(eye1,  focus, 0.004f * distY * ROTATE_STEP) ;
          RotateRight(eye1,  focus, 0.004f * distX * ROTATE_STEP) ;

          mouseGrabLastX = x;
          mouseGrabLastY = y;

          update = true;

          UpdateCamera(focus, eye1);

          glutPostRedisplay();
          lastMouseUpdate = WallClockTime();
        }
    }
}

void reshape(int w, int h) {

  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH rootNode = cntxt->getRootNode();
  gvt::core::DBNodeH filmNode = rootNode["Film"];

  width=w;
  height=h;
  filmNode["width"]=w;
  filmNode["height"]=h;

  update = true;

  glViewport(0, 0, (GLsizei)w, (GLsizei)h);

  if (renderMode == FILM_RENDER_MODE){

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

    }
  else{
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluPerspective(45,(double)w/(double)h,.1,100);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      Point4f eye1 = gvt::core::variant_toPoint4f(camNode["eyePoint"].value());
      Point4f focus = gvt::core::variant_toPoint4f(camNode["focus"].value());
      Point4f up = gvt::core::variant_toVector4f(camNode["upVector"].value());

      gluLookAt(eye1[0],eye1[1],eye1[2],
          focus[0],focus[1],focus[2],
          up[0],up[1],up[2]);
    }
}

void drawWireBox(gvt::render::data::primitives::Box3D& bbox ) {
  float xmin= bbox.bounds[0][0];
  float ymin= bbox.bounds[0][1];
  float zmin= bbox.bounds[0][2];
  float xmax= bbox.bounds[1][0];
  float ymax= bbox.bounds[1][1];
  float zmax= bbox.bounds[1][2];

  glPushMatrix();
  glTranslatef(0.5f*(xmin + xmax), 0.5f*(ymin + ymax), 0.5f*(zmin + zmax));
  glScalef(xmax - xmin, ymax - ymin, zmax - zmin);
  glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
  glutWireCube(1.0f);
  glPopMatrix();
}

void RenderBVH(){

  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH rootNode = cntxt->getRootNode();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);

  for ( gvt::core::DBNodeH instance : rootNode["Instances"].getChildren()){
      Box3D *bbox = gvt::core::variant_toBox3DPtr(instance["bbox"].value());
      drawWireBox(*bbox);
    }

  //glutSolidTeapot(.5);

  glFlush();

}

void Render() {
  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH rootNode = cntxt->getRootNode();
  gvt::core::DBNodeH camNode = rootNode["Camera"];
  gvt::core::DBNodeH filmNode = rootNode["Film"];

  // setup gvtCamera from database entries
  gvtPerspectiveCamera mycamera;
  Vector4f up = gvt::core::variant_toVector4f(camNode["upVector"].value());
  Point4f focus = gvt::core::variant_toPoint4f(camNode["focus"].value());
  Point4f cameraposition = gvt::core::variant_toPoint4f(camNode["eyePoint"].value());

  float fov = gvt::core::variant_toFloat(camNode["fov"].value());
  mycamera.lookAt(cameraposition,focus,up);
  mycamera.setFOV(fov);
  mycamera.setFilmsize(gvt::core::variant_toInteger(
                         filmNode["width"].value()),
      gvt::core::variant_toInteger(filmNode["height"].value()));

  int schedType = gvt::core::variant_toInteger(rootNode["Schedule"]["type"].value());
  switch(schedType) {
    case gvt::render::scheduler::Image :
      {
        std::cout << "starting image scheduler" << std::endl;
        for(int z=0; z<1; z++) {
            mycamera.AllocateCameraRays();
            mycamera.generateRays();
            gvt::render::algorithm::Tracer<ImageScheduler>(mycamera.rays,*imageptr)();
          }
        break;
      }
    case gvt::render::scheduler::Domain :
      {
        std::cout << "starting domain scheduler" << std::endl;
        gvt::render::algorithm::Tracer<DomainScheduler>(mycamera.rays,*imageptr)();
        break;
      }
    default:
      {
        std::cout << "unknown schedule type provided: " << schedType << std::endl;
        break;
      }
    }

}

void RenderFilm(){
  unsigned char key = 'r';
  if (update) {
      MPI_Bcast(&key, 1, MPI_UNSIGNED_CHAR, master, MPI_COMM_WORLD);
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
    case BVH_RENDER_MODE:
      {
        RenderBVH();
        break;
      }
    case FILM_RENDER_MODE:
      {
        RenderFilm();
        break;
      }
    default:
      {
        cout << "unknown render mode" << endl;
        break;
      }
    }
}

void specialkey(int key, int x, int y) {

  Point4f eye1 = gvt::core::variant_toPoint4f(camNode["eyePoint"].value());
  Point4f focus = gvt::core::variant_toPoint4f(camNode["focus"].value());


  switch (key) {
    case GLUT_KEY_LEFT: // translate camera left
      TranslateLeft(eye1, focus, MOVE_STEP);
      update = true;
      break;
    case GLUT_KEY_RIGHT: // translate camera right
      TranslateRight(eye1, focus, MOVE_STEP);
      update = true;
      break;
    case GLUT_KEY_UP: // translate camera right
      TranslateForward(eye1, focus, MOVE_STEP);
      update = true;
      break;
    case GLUT_KEY_DOWN: // translate camera right
      TranslateBackward(eye1, focus, MOVE_STEP);
      update = true;
      break;
    default:
      break;
    }

  UpdateCamera(focus, eye1);

  glutPostRedisplay();
}

void UpdateRenderMode()
{
  if (renderMode == BVH_RENDER_MODE){
      renderMode = FILM_RENDER_MODE;

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

    }
  else{
      renderMode = BVH_RENDER_MODE;
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluPerspective(45,(double)width/(double)height,.1,100);
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      Point4f eye1 = gvt::core::variant_toPoint4f(camNode["eyePoint"].value());
      Point4f focus = gvt::core::variant_toPoint4f(camNode["focus"].value());
      Point4f up = gvt::core::variant_toVector4f(camNode["upVector"].value());

      gluLookAt(eye1[0],eye1[1],eye1[2],
          focus[0],focus[1],focus[2],
          up[0],up[1],up[2]);
    }
}

void keyboard(unsigned char key, int x, int y) {

  Point4f eye1 = gvt::core::variant_toPoint4f(camNode["eyePoint"].value());
  Point4f focus = gvt::core::variant_toPoint4f(camNode["focus"].value());

  switch (key) {
    case ESCAPE:
      MPI_Bcast(&key, 1, MPI_UNSIGNED_CHAR, master, MPI_COMM_WORLD);
      if (MPI::COMM_WORLD.Get_size() > 1)
        MPI_Finalize();
      exit(0);

    case 'q':
      MPI_Bcast(&key, 1, MPI_UNSIGNED_CHAR, master, MPI_COMM_WORLD);
      if (MPI::COMM_WORLD.Get_size() > 1)
        MPI_Finalize();
      exit(0);

    case 'w':
      RotateUp(eye1,  focus,  ROTATE_STEP) ;
      update = true;
      break;
    case 's':
      RotateDown(eye1,  focus,  ROTATE_STEP) ;
      update = true;
      break;
    case 'a':
      RotateLeft(eye1,  focus,  ROTATE_STEP) ;
      update = true;
      break;
    case 'd':
      RotateRight(eye1,  focus,  ROTATE_STEP) ;
      update = true;
      break;

    case 'm':
      UpdateRenderMode();
      break;
    default:
      // dont do anything
      break;
    }

  if (update)
    UpdateCamera(focus, eye1);

  glutPostRedisplay();
}

void ConfigSceneFromFile(string filename){

  gvtapps::render::ConfigFileLoader cl(filename);
  gvt::render::data::Dataset& scene = cl.scene;

  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();

  gvt::core::DBNodeH root = cntxt->getRootNode();
  gvt::core::DBNodeH dataNodes = cntxt->createNodeFromType("Data","Data",root.UUID());
  gvt::core::DBNodeH instNodes =
      cntxt->createNodeFromType("Instances", "Instances", root.UUID());

  for (int i=0; i< scene.domainSet.size(); i++){

      Mesh* mesh = ((GeometryDomain*)scene.getDomain(i))->getMesh();

      gvt::core::DBNodeH meshNode = cntxt->createNodeFromType("Mesh",
                                                              filename.c_str(), dataNodes.UUID());

      meshNode["file"] = filename;
      //mesh->computeBoundingBox();
      gvt::render::data::primitives::Box3D * bbox = mesh->getBoundingBox();
      meshNode["bbox"] = bbox;
      meshNode["ptr"] = mesh;


      // add instance
      gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance","inst",
                                                              instNodes.UUID());
      Box3D *mbox = gvt::core::variant_toBox3DPtr(meshNode["bbox"].value());
      instnode["id"] = i;
      instnode["meshRef"]=meshNode.UUID();
      auto m = new gvt::core::math::AffineTransformMatrix<float>(true);
      auto minv = new gvt::core::math::AffineTransformMatrix<float>(true);
      auto normi = new gvt::core::math::Matrix3f();
      instnode["mat"] = m;
      *minv = m->inverse();
      instnode["matInv"] = minv;
      *normi = m->upper33().inverse().transpose();
      instnode["normi"] = normi;
      auto il = (*m) * mbox->bounds[0];
      auto ih = (*m) * mbox->bounds[1];
      Box3D *ibox = new gvt::render::data::primitives::Box3D(il,ih);
      instnode["bbox"] = ibox;
      instnode["centroid"] = ibox->centroid();

    }

  // add lights, camera, and film to the database
  gvt::core::DBNodeH lightNodes = cntxt->createNodeFromType("Lights", "Lights", root.UUID());
  gvt::core::DBNodeH lightNode = cntxt->createNodeFromType("PointLight", "PointLight", lightNodes.UUID());
  gvt::render::data::scene::PointLight * lp=
      (gvt::render::data::scene::PointLight *)scene.getLight(0);
  lightNode["position"] = Vector4f(lp->position);
  lightNode["color"] = Vector4f(lp->color);

  // camera
  gvt::core::DBNodeH _camNode = cntxt->createNodeFromType("Camera","Camera",root.UUID());
  _camNode["eyePoint"] = Point4f(scene.camera.getEye());
  _camNode["focus"] = Point4f(scene.camera.getFocus());
  _camNode["upVector"] = scene.camera.getUp();
  _camNode["fov"] = (float)(45.0 * M_PI/180.0); // TODO

  // film
  gvt::core::DBNodeH filmNode = cntxt->createNodeFromType("Film","Film",root.UUID());
  filmNode["width"] = int(scene.camera.getFilmSizeWidth());
  filmNode["height"] = int(scene.camera.getFilmSizeHeight());

}

void ConfigSceneCubeCone(){

  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();

  gvt::core::DBNodeH root = cntxt->getRootNode();
  // mix of cones and cubes

  // TODO: maybe rename to 'Data' - as it can store different types of data
  // [mesh, volume, lines]
  gvt::core::DBNodeH dataNodes =
      cntxt->createNodeFromType("Data", "Data", root.UUID());

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
    coneMeshNode["bbox"] = meshbbox;
    coneMeshNode["ptr"] = mesh;
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
    cubeMeshNode["bbox"] = meshbbox;
    cubeMeshNode["ptr"] = mesh;
  }

  gvt::core::DBNodeH instNodes =
      cntxt->createNodeFromType("Instances", "Instances", root.UUID());

  // create a NxM grid of alternating cones / cubes, offset using i and j
  int instId = 0;
  int ii[2] = { -2, 3 }; // i range
  int jj[2] = { -2, 3 }; // j range
  for (int i = ii[0]; i < ii[1]; i++) {
      for (int j = jj[0]; j < jj[1]; j++) {

          gvt::core::DBNodeH instnode =
              cntxt->createNodeFromType("Instance", "inst", instNodes.UUID());
          // gvt::core::DBNodeH meshNode = (instId % 2) ? coneMeshNode :
          // cubeMeshNode;
          gvt::core::DBNodeH meshNode = (instId % 2) ? cubeMeshNode : coneMeshNode;
          Box3D *mbox = gvt::core::variant_toBox3DPtr(meshNode["bbox"].value());

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

          instnode["mat"] = m;
          *minv = m->inverse();
          instnode["matInv"] = minv;
          *normi = m->upper33().inverse().transpose();
          instnode["normi"] = normi;

          auto il = (*m) * mbox->bounds[0];
          auto ih = (*m) * mbox->bounds[1];
          Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
          instnode["bbox"] = ibox;
          instnode["centroid"] = ibox->centroid();
        }
    }

  // add lights, camera, and film to the database
  gvt::core::DBNodeH lightNodes =
      cntxt->createNodeFromType("Lights", "Lights", root.UUID());
  gvt::core::DBNodeH lightNode =
      cntxt->createNodeFromType("PointLight", "PointLight", lightNodes.UUID());
  lightNode["position"] = Vector4f(1.0, 0.0, 0.0, 0.0);
  lightNode["color"] = Vector4f(1.0, 1.0, 1.0, 0.0);

  //  // second light just for fun
  //  gvt::core::DBNodeH lN2 = cntxt->createNodeFromType("PointLight",
  //                                                     "conelight", lightNodes.UUID());
  //  lN2["position"] = Vector4f(2.0, 2.0, 2.0, 0.0);
  //  lN2["color"] = Vector4f(0.0, 0.0, 0.0, 0.0);

  gvt::core::DBNodeH _camNode =
      cntxt->createNodeFromType("Camera", "Camera", root.UUID());
  _camNode["eyePoint"] = Point4f(4.0, 0.0, 0.0, 1.0);
  _camNode["focus"] = Point4f(0.0, 0.0, 0.0, 1.0);
  _camNode["upVector"] = Vector4f(0.0, 1.0, 0.0, 0.0);
  _camNode["fov"] = (float)(45.0 * M_PI / 180.0);

  gvt::core::DBNodeH filmNode =
      cntxt->createNodeFromType("Film", "Film", root.UUID());
  filmNode["width"] = 512;
  filmNode["height"] = 512;
}

int main(int argc, char *argv[]) {
  unsigned char action;
  // mpi initialization

  int rank = -1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Barrier(MPI_COMM_WORLD);
  master = MPI::COMM_WORLD.Get_size() - 1;

  string filename;

  if (argc > 1) {
      filename = argv[1];
    } else {
      cerr << " application requires input config file" << endl;
      if (MPI::COMM_WORLD.Get_size() > 1)
        MPI_Finalize();
      exit(1);
    }

  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();

  if (cntxt == NULL) {
      std::cout << "Something went wrong initializing the context" << std::endl;
      exit(0);
    }

  ConfigSceneFromFile(filename);
  //ConfigSceneCubeCone();

  gvt::core::DBNodeH root = cntxt->getRootNode();


  // TODO: schedule db design could be modified a bit
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
  width =  gvt::core::variant_toInteger(root["Film"]["width"].value());
  height=  gvt::core::variant_toInteger(root["Film"]["height"].value());

  Image image(width,
              height, "spoot");

  imageptr = &image;
  imagebuffer = image.GetBuffer();

  Render();

  if (rank == master) { // max rank process does display

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

      if (renderMode == FILM_RENDER_MODE){

          glMatrixMode(GL_PROJECTION);
          glLoadIdentity();
          gluOrtho2D(0.0, (GLdouble)width, 0.0, (GLdouble)height);
          glMatrixMode(GL_MODELVIEW);
          glLoadIdentity();

        }
      else{
          glMatrixMode(GL_PROJECTION);
          glLoadIdentity();
          gluPerspective(45,(double)width/(double)height,.1,100);
          glMatrixMode(GL_MODELVIEW);
          glLoadIdentity();

          Point4f eye1 = gvt::core::variant_toPoint4f(camNode["eyePoint"].value());
          Point4f focus = gvt::core::variant_toPoint4f(camNode["focus"].value());
          Point4f up = gvt::core::variant_toVector4f(camNode["upVector"].value());

          gluLookAt(eye1[0],eye1[1],eye1[2],
              focus[0],focus[1],focus[2],
              up[0],up[1],up[2]);
        }

      glutMainLoop();
    } else { // loop and wait for further instructions
      while (1) {
          MPI_Bcast(&action, 1, MPI_CHAR, master, MPI_COMM_WORLD);
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
            default:
              break;
            }
        }
    }
  return 0;
}
