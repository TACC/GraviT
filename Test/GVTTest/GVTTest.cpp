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
/**
 * A simple GraviT application to do some testing.
 * This is supposed to be as close to the osptest application as I can get it. 
 *
*/
#include <gvt/render/RenderContext.h>
#include <gvt/render/Types.h>
#include <vector>
#include <algorithm>
#include <set>
#include <gvt/core/mpi/Wrapper.h>
#include <gvt/core/Math.h>
#include <gvt/render/data/Dataset.h>
#include <gvt/render/data/Domains.h>
#include <gvt/render/Schedulers.h>

#ifdef GVT_RENDER_ADAPTER_EMBREE
#include <gvt/render/adapter/embree/Wrapper.h>
#endif

#ifdef GVT_RENDER_ADAPTER_MANTA
#include <gvt/render/adapter/manta/Wrapper.h>
#endif

#ifdef GVT_RENDER_ADAPTER_OPTIX
#include <gvt/render/adapter/optix/Wrapper.h>
#endif

#ifdef GVT_USE_MPE
#include "mpe.h"
#endif
#include <gvt/render/algorithm/Tracers.h>
#include <gvt/render/data/scene/gvtCamera.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/data/Primitives.h>

#include <boost/range/algorithm.hpp>

#include <math.h>
#include <stdio.h>
#include "../iostuff.h"
#include "../timer.h"

using namespace std;
using namespace gvt::render;
using namespace gvt::core::math;
using namespace gvt::core::mpi;
using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;
using namespace gvt::render::data::primitives;
static Vertex **vlist;
static Face **flist;

#define MIN(a, b) ((a < b) ? (a) : (b))
#define MAX(a, b) ((a > b) ? (a) : (b))
void findbounds(float* array, int numelements,  Point4f* lower,Point4f* upper)
{
  float xmin,xmax,ymin,ymax,zmin,zmax;
  xmin = array[0];
  ymin = array[1];
  zmin = array[2];
  xmax = xmin;
  ymax = ymin;
  zmax = zmin;
  for(int i=0;i<numelements;i++) 
  {
    xmin = MIN(array[3*i], xmin);
    ymin = MIN(array[3*i+1], ymin);
    zmin = MIN(array[3*i+2], zmin);
    xmax = MAX(array[3*i], xmax);
    ymax = MAX(array[3*i+1], ymax);
    zmax = MAX(array[3*i+2], zmax);
  }
  *lower = Point4f(xmin, ymin, zmin);
  *upper = Point4f(xmax, ymax, zmax);
}

int main(int argc, char **argv) {

  // default values
  int width = 1920;
  int height = 1080;
  int warmupframes = 1;
  int benchmarkframes = 10;
  // timer stuff
  my_timer_t startTime, endTime;
  double rendertime = 0.0;
  double iotime = 0.0;
  double modeltime = 0.0;
  // geometry data
  float *vertexarray;
  float *colorarray;
  int32_t *indexarray;
  int numtriangles = 0;
  int nverts,nfaces;
  // file related things
  string filepath("");
  string filename("");
  string outputfile("");
  string renderertype("obj");
  // initialize gravit context
  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  if (cntxt == NULL) {
    std::cout << "Something went wrong initializing the context" << std::endl;
    exit(0);
  }
  gvt::core::DBNodeH root = cntxt->getRootNode();
  gvt::core::DBNodeH dataNodes = cntxt->createNodeFromType("Data", "Data", root.UUID());
  gvt::core::DBNodeH instNodes = cntxt->createNodeFromType("Instances", "Instances", root.UUID());

  // parse the command line
  if ((argc <2)) 
  {
    // no input so render the defalut empty image
  } 
  else
  {
    // parse the input 
    for(int i=1;i<argc;i++)
    {
      const string arg = argv[i];
      if(arg == "-i" )
      {
        filepath = argv[++i];
        if(!file_exists(filepath.c_str())) 
        {
          cout << "File \"" << filepath << "\" does not exist. Exiting." << endl;
          return 0;
        }
        else if (isdir(filepath.c_str()))
        {
          vector<string> files = findply(filepath);
          if(!files.empty()) // directory contains .ply files
          {
            vector<string>::const_iterator file;
            int k;
            char txt[16];
            for(file=files.begin(),k=0;file!=files.end();file++,k++)
            {
              timeCurrent(&startTime);
              ReadPlyData(*file,vertexarray,colorarray,indexarray,nverts,nfaces);
              timeCurrent(&endTime);
              iotime += timeDifferenceMS(&startTime,&endTime);

              timeCurrent(&startTime);
              sprintf(txt, "%d", k);
              filename = "block";
              filename += txt;
              gvt::core::DBNodeH EnzoMeshNode = cntxt->createNodeFromType("Mesh", *file, dataNodes.UUID());
              Mesh *mesh = new Mesh(new Lambert(Vector4f(1.0, 1.0, 1.0, 1.0)));
              for (int i = 0; i < nverts; i++) 
              {
                mesh->addVertex(Point4f(vertexarray[3*i], vertexarray[3*i+1], vertexarray[3*i+2], 1.0));
              }
              for (int i = 0; i < nfaces; i++) //Add faces to mesh
              {
                mesh->addFace(indexarray[3*i]+1,indexarray[3*i+1]+1,indexarray[3*i+2]+1);
              }
              Point4f lower;
              Point4f upper;
              findbounds(vertexarray,nverts,&lower,&upper);
              Box3D *meshbbox = new gvt::render::data::primitives::Box3D(lower, upper);
              mesh->generateNormals();
              EnzoMeshNode["file"] = filename;
              EnzoMeshNode["bbox"] = meshbbox;
              EnzoMeshNode["ptr"] = mesh;
              // add instance
              gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance", "inst", instNodes.UUID());
              gvt::core::DBNodeH meshNode = EnzoMeshNode;
              Box3D *mbox = gvt::core::variant_toBox3DPtr(meshNode["bbox"].value());
              instnode["id"] = k;
              instnode["meshRef"] = meshNode.UUID();
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
              Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
              instnode["bbox"] = ibox;
              instnode["centroid"] = ibox->centroid();
              timeCurrent(&endTime);
              modeltime+=timeDifferenceMS(&startTime,&endTime);
              numtriangles += nfaces;
            }
          }
          else // directory has no .ply files
          {
            filepath="";
          }
        }
        else // filepath is not a directory but a .ply file
        {
          timeCurrent(&startTime);
          ReadPlyData(filepath,vertexarray,colorarray,indexarray,nverts,nfaces);
          timeCurrent(&endTime);
          iotime+=timeDifferenceMS(&startTime,&endTime);
          gvt::core::DBNodeH EnzoMeshNode = cntxt->createNodeFromType("Mesh", filepath.c_str(), dataNodes.UUID());
          numtriangles += nfaces;
        }
      }
    }
  }
  std::string   rootdir;
  rootdir = "/work/01197/semeraro/maverick/DAVEDATA/EnzoPlyData/";
#if 1
  MPI_Init(&argc, &argv);
  MPI_Pcontrol(0);
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif


  timeCurrent(&startTime);
  // add lights, camera, and film to the database
  gvt::core::DBNodeH lightNodes = cntxt->createNodeFromType("Lights", "Lights", root.UUID());
  gvt::core::DBNodeH lightNode = cntxt->createNodeFromType("PointLight", "conelight", lightNodes.UUID());
  lightNode["position"] = Vector4f(512.0, 512.0, 2048.0, 0.0);
  lightNode["color"] = Vector4f(1.0, 1.0, 1.0, 0.0);
  // camera
  gvt::core::DBNodeH camNode = cntxt->createNodeFromType("Camera", "conecam", root.UUID());
  camNode["eyePoint"] = Point4f(512.0, 512.0, 4096.0, 1.0);
  camNode["focus"] = Point4f(512.0, 512.0, 0.0, 1.0);
  camNode["upVector"] = Vector4f(0.0, 1.0, 0.0, 0.0);
  camNode["fov"] = (float)(25.0 * M_PI / 180.0);
  // film
  gvt::core::DBNodeH filmNode = cntxt->createNodeFromType("Film", "conefilm", root.UUID());
  filmNode["width"] = width;
  filmNode["height"] = height;

  gvt::core::DBNodeH schedNode = cntxt->createNodeFromType("Schedule", "Enzosched", root.UUID());
  schedNode["type"] = gvt::render::scheduler::Image;
// schedNode["type"] = gvt::render::scheduler::Domain;

#ifdef GVT_RENDER_ADAPTER_EMBREE
  int adapterType = gvt::render::adapter::Embree;
#elif GVT_RENDER_ADAPTER_MANTA
  int adapterType = gvt::render::adapter::Manta;
#elif GVT_RENDER_ADAPTER_OPTIX
  int adapterType = gvt::render::adapter::Optix;
#elif
  GVT_DEBUG(DBG_ALWAYS, "ERROR: missing valid adapter");
#endif

  schedNode["adapter"] = gvt::render::adapter::Embree;

  // end db setup

  // use db to create structs needed by system

  // setup gvtCamera from database entries
  gvtPerspectiveCamera mycamera;
  Point4f cameraposition = gvt::core::variant_toPoint4f(camNode["eyePoint"].value());
  Point4f focus = gvt::core::variant_toPoint4f(camNode["focus"].value());
  float fov = gvt::core::variant_toFloat(camNode["fov"].value());
  Vector4f up = gvt::core::variant_toVector4f(camNode["upVector"].value());
  mycamera.lookAt(cameraposition, focus, up);
  mycamera.setFOV(fov);
  mycamera.setFilmsize(gvt::core::variant_toInteger(filmNode["width"].value()),
                       gvt::core::variant_toInteger(filmNode["height"].value()));

#ifdef GVT_USE_MPE
  MPE_Log_event(readend, 0, NULL);
#endif
  // setup image from database sizes
  Image myimage(mycamera.getFilmSizeWidth(), mycamera.getFilmSizeHeight(), "enzo");

  mycamera.AllocateCameraRays();
  mycamera.generateRays();
  timeCurrent(&endTime);
  modeltime+=timeDifferenceMS(&startTime,&endTime);

  int schedType = gvt::core::variant_toInteger(root["Schedule"]["type"].value());
  switch (schedType) {
  case gvt::render::scheduler::Image: {
    std::cout << "starting image scheduler" << std::endl;
    timeCurrent(&startTime);
    for (int z = 0; z < 1; z++) {
  //    mycamera.AllocateCameraRays();
   //   mycamera.generateRays();
      gvt::render::algorithm::Tracer<ImageScheduler>(mycamera.rays, myimage)();
    }
    timeCurrent(&endTime);
    rendertime += timeDifferenceMS(&startTime,&endTime);
    break;
  }
  case gvt::render::scheduler::Domain: {
    std::cout << "starting domain scheduler" << std::endl;
    timeCurrent(&startTime);
    gvt::render::algorithm::Tracer<DomainScheduler>(mycamera.rays, myimage)();
    timeCurrent(&endTime);
    rendertime += timeDifferenceMS(&startTime,&endTime);
    break;
  }
  default: {
    std::cout << "unknown schedule type provided: " << schedType << std::endl;
    break;
  }
  }

  myimage.Write();
#ifdef GVT_USE_MPI
  if (MPI::COMM_WORLD.Get_size() > 1)
    MPI_Finalize();
#endif
}

