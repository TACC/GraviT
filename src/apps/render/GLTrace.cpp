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

#include <GL/freeglut.h>
#define ESCAPE 27

using namespace std;
using namespace gvtapps::render;
using namespace gvt::render::adapter::manta::data::domain;
using namespace gvt::render::data::domain;
using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;
using namespace gvt::core::mpi;
using namespace gvt::core::math;

// global variables used by glut callbacks.
//
GLubyte *imagebuffer;
static GLint width, height;
static GLint Height;
Image *imageptr;
gvt::render::actor::RayVector rays;
gvt::render::data::Dataset *sceneptr;
int master;
bool update = false;

// ************************* ray tracing functions
// **********************************************

void Render() {
  rays = sceneptr->camera.MakeCameraRays();
  gvt::render::algorithm::Tracer<DomainScheduler>(rays, (*imageptr))();
}

// ************************* Glut callback  functions
// ********************************************

void dispfunc(void) {
  unsigned char key = 'r';
  if (update) {
    MPI_Bcast(&key, 1, MPI_UNSIGNED_CHAR, master, MPI_COMM_WORLD);
    Render();
    update = false;
  }
  glClear(GL_COLOR_BUFFER_BIT);
  glRasterPos2i(0, 0);
  glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, imagebuffer);
  glFlush();
}

void reshape(int w, int h) {
  glViewport(0, 0, (GLsizei)w, (GLsizei)h);
  Height = (GLint)h;
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void specialkey(int key, int x, int y) {
  Vector4f eye1, focus, up1;
  eye1 = sceneptr->camera.getEye();
  up1 = sceneptr->camera.getUp();
  switch (key) {
  case GLUT_KEY_LEFT: // translate camera left
    eye1[0] = eye1[0] - 0.05;
    update = true;
    break;
  case GLUT_KEY_RIGHT: // translate camera right
    eye1[0] = eye1[0] + 0.05;
    update = true;
    break;
  case GLUT_KEY_UP: // translate camera right
    eye1[1] = eye1[1] + 0.05;
    update = true;
    break;
  case GLUT_KEY_DOWN: // translate camera right
    eye1[1] = eye1[1] - 0.05;
    update = true;
    break;
  default:
    break;
  }
  sceneptr->camera.setEye(eye1);
  glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
  switch (key) {
  case ESCAPE:
    MPI_Bcast(&key, 1, MPI_UNSIGNED_CHAR, master, MPI_COMM_WORLD);
    if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();
    exit(0);
  case 'q':
    MPI_Bcast(&key, 1, MPI_UNSIGNED_CHAR, master, MPI_COMM_WORLD);
    if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();
    exit(0);
  default:
    // dont do anything
    break;
  }
}

// ******************************* main
// ****************************************************
int main(int argc, char *argv[]) {
  unsigned char action;
  // mpi initialization

  int rank = -1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Barrier(MPI_COMM_WORLD);
  master = MPI::COMM_WORLD.Get_size() - 1;

  // input

  string filename;

  if (argc > 1) {
    filename = argv[1];
  } else {
    cerr << " application requires input config file" << endl;
    if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();
    exit(1);
  }

  gvtapps::render::ConfigFileLoader cl(filename);
  //	GVT::Frontend::ConfigFileLoader cl(filename);
  //
  // 	Do Ray Tracing without the explict use of the RayTracer class.
  // 	Because RayTracer.RenderImage() writes a file and I dont want to
  // 	change that class.
  //
  gvt::render::data::Dataset *scene(&cl.scene);
  sceneptr = scene;
  scene->camera.SetCamera(rays, 1.0);
  gvt::render::Attributes &rta = *(gvt::render::Attributes::instance());
  rta.dataset = new gvt::render::data::Dataset();

  // clang-format off
  BOOST_FOREACH(AbstractDomain * dom, scene->domainSet) {
    GeometryDomain *d = (GeometryDomain *)dom;
    d->setLights(scene->lightSet);
    rta.dataset->addDomain(new MantaDomain(d));
  }
  // clang-forat on

  rta.view.width = scene->camera.getFilmSizeWidth();
  rta.view.height = scene->camera.getFilmSizeHeight();
  rta.view.camera = scene->camera.getEye();
  rta.view.focus = scene->camera.getLook();
  rta.view.up = scene->camera.up;
  rta.do_lighting = true;
  rta.schedule = gvt::render::Attributes::Image;
  rta.render_type = gvt::render::Attributes::Manta;

  Image image(scene->camera.getFilmSizeWidth(), scene->camera.getFilmSizeHeight(), "spoot");
  imageptr = &image;
  imagebuffer = image.GetBuffer();
  width = rta.view.width;
  height = rta.view.height;
  Render();

  //
  // Opengl display stuff goes here
  //

  if (rank == master) { // max rank process does display
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(width, height);
    glutInitWindowPosition(10, 10);
    glutCreateWindow(filename.c_str());
    glClearColor(0., 0., 0., 0.);
    glShadeModel(GL_FLAT);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glutDisplayFunc(dispfunc);
    glutSpecialFunc(specialkey);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
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
