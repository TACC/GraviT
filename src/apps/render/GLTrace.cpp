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



#define ESCAPE 27

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
static GLint width, height;
static GLint Height;
Image *imageptr;
//gvt::render::actor::RayVector rays;
//gvt::render::data::Dataset *sceneptr;
int master;
bool update = false;

// ************************* ray tracing functions
// **********************************************

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

  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  gvt::core::DBNodeH rootNode = cntxt->getRootNode();
  gvt::core::DBNodeH camNode = rootNode["Camera"];
  Point4f eye1 = gvt::core::variant_toPoint4f(camNode["eyePoint"].value());


  //Vector4f eye1, focus, up1;
  //eye1 = sceneptr->camera.getEye();
  //up1 = sceneptr->camera.getUp();

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

  //sceneptr->camera.setEye(eye1);

  camNode["eyePoint"] = eye1;
  glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
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

  string filename;

  if (argc > 1) {
      filename = argv[1];
    } else {
      cerr << " application requires input config file" << endl;
      if (MPI::COMM_WORLD.Get_size() > 1)
        MPI_Finalize();
      exit(1);
    }

  gvtapps::render::ConfigFileLoader cl(filename);
  gvt::render::data::Dataset& scene = cl.scene;


  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  if (cntxt == NULL) {
      std::cout << "Something went wrong initializing the context" << std::endl;
      exit(0);
    }

  gvt::core::DBNodeH root = cntxt->getRootNode();
  gvt::core::DBNodeH dataNodes = cntxt->createNodeFromType("Data","Data",root.UUID());
  gvt::core::DBNodeH instNodes =
      cntxt->createNodeFromType("Instances", "Instances", root.UUID());

  for (int i; i< scene.domainSet.size(); i++){

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
  gvt::core::DBNodeH camNode = cntxt->createNodeFromType("Camera","Camera",root.UUID());
  camNode["eyePoint"] = Point4f(scene.camera.getEye());
  camNode["focus"] = Point4f(scene.camera.getFocus());
  camNode["upVector"] = scene.camera.getUp();
  camNode["fov"] = (float)(25.0 * M_PI/180.0); // TODO

  // film
  gvt::core::DBNodeH filmNode = cntxt->createNodeFromType("Film","Film",root.UUID());
  filmNode["width"] = int(scene.camera.getFilmSizeWidth());
  filmNode["height"] = int(scene.camera.getFilmSizeHeight());


  //Scheduler
  gvt::core::DBNodeH schedNode = cntxt->createNodeFromType("Schedule","Schedule",root.UUID());
  schedNode["type"] = gvt::render::scheduler::Image;
  //schedNode["type"] = gvt::render::scheduler::Domain;

#ifdef GVT_RENDER_ADAPTER_EMBREE
  int adapterType = gvt::render::adapter::Embree;
#elif GVT_RENDER_ADAPTER_MANTA
  int adapterType = gvt::render::adapter::Manta;
#elif GVT_RENDER_ADAPTER_OPTIX
  int adapterType = gvt::render::adapter::Optix;
#elif
  GVT_DEBUG(DBG_ALWAYS, "ERROR: missing valid adapter");
#endif

  schedNode["adapter"] = gvt::render::adapter::Manta;

  //	GVT::Frontend::ConfigFileLoader cl(filename);
  //
  // 	Do Ray Tracing without the explict use of the RayTracer class.
  // 	Because RayTracer.RenderImage() writes a file and I dont want to
  // 	change that class.
  //
  //  gvt::render::data::Dataset *scene(&cl.scene);
  //  sceneptr = scene;
  //  scene->camera.SetCamera(rays, 1.0);

  //  gvt::render::Attributes &rta = *(gvt::render::Attributes::instance());
  //  rta.dataset = new gvt::render::data::Dataset();

  //  BOOST_FOREACH (AbstractDomain *dom, scene->domainSet) {
  //    GeometryDomain *d = (GeometryDomain *)dom;
  //    d->setLights(scene->lightSet);
  //    rta.dataset->addDomain(new MantaDomain(d));
  //  }

  //  rta.view.width = scene->camera.getFilmSizeWidth();
  //  rta.view.height = scene->camera.getFilmSizeHeight();
  //  rta.view.camera = scene->camera.getEye();
  //  rta.view.focus = scene->camera.getLook();
  //  rta.view.up = scene->camera.up;
  //  rta.do_lighting = true;
  //  rta.schedule = gvt::render::Attributes::Image;
  //  rta.render_type = gvt::render::Attributes::Manta;



  Image image(scene.camera.getFilmSizeWidth(),
              scene.camera.getFilmSizeHeight(), "spoot");

  imageptr = &image;

  imagebuffer = image.GetBuffer();

  width = scene.camera.getFilmSizeWidth();
  height = scene.camera.getFilmSizeHeight();


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
