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
 * A simple GraviT application that loads some volume data and renders it.
 *
 * GraviT volume rendering test. Reads a Block of Values (bov) file, a transfer function
 * file and performs volume rendering using the OSPRay adapter. As of this writing the 
 * only adapter capable of volume rendering is the OSPRay adapter. 
 *
*/
#include <algorithm>
#include <gvt/core/Math.h>
#include <gvt/core/mpi/Wrapper.h>
#include <gvt/render/RenderContext.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/Types.h>
#include <gvt/render/data/Domains.h>
#include <set>
#include <vector>

#include <tbb/task_scheduler_init.h>
#include <thread>

#ifdef GVT_RENDER_ADAPTER_OSPRAY
#include <gvt/render/adapter/ospray/Wrapper.h>
#endif

#ifdef GVT_USE_MPE
#include "mpe.h"
#endif
#include <gvt/render/algorithm/Tracers.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/data/scene/gvtCamera.h>

#include <boost/range/algorithm.hpp>

#include <iostream>
#include <fstream>
#include <ios>
#include <math.h>
#include <glob.h>
#include <sys/stat.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>

#include "ParseCommandLine.h"

using namespace std;
//using namespace gvt::render;
//using namespace gvt::core::mpi;
//using namespace gvt::render::data::scene;
//using namespace gvt::render::schedule;
//using namespace gvt::render::data::primitives;

std::vector<std::string> split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss,item,delim)) {
      if(item.length() > 0) {
        elems.push_back(item);
      }
    }
    return elems;
}
// the bovheader struct reads a bov file header. Other fcns read the data
// the data is mapped to a float for return 
// to the caller. 
struct bovheader { 
  std::ifstream myfile;
  std::string datafile;
  std::string headerfile;
  std::string headerdir;
  // the number of points in each coordinate direction
  // in the global mesh
  int datasize[3];
  int numberofdomains;
  // number of partitions of the global 
  // index space in each direction. 
  int xpartitions, ypartitions, zpartitions;
  enum dataformat {
    INT,
    FLOAT,
    UINT,
    SHORT
  } dfmt;
  std::string variable;
  enum endian {
    BIG,
    LITTLE
  } dendian;
  bool dividebrick ;
  // the number of points in each coordinate direction 
  // in the local block. (read from header)
  int bricklets[3];
  // the number of points in the local block including overlap
  // overlap one point in each direction
  int counts[3];
  float origin[3];
  gvt::render::data::primitives::Box3D *volbox;
  bovheader(std::string headername): headerfile(headername) {
    for(int i=0;i<3;i++) {
      datasize[i] = 0;
      bricklets[i] = 1;
    }
    myfile.open(headerfile.c_str());
    headerdir=headerfile.substr(0,headerfile.find_last_of("/\\"));
    std::string line;
    while(myfile.good()) {
      std::vector<std::string> elems;
      std::getline(myfile,line);
      if(!line.empty()) {
        split(line,' ',elems);
        if(elems[0] == "DATA_FILE:") {
          // build a full path name for the file
          // assume the file name in the header is
          // relative to the location of the header itself
          // concatinate the directory of the header file
          // to the data_file name.
          datafile = headerdir + "/" + elems[1];
        } else if(elems[0] == "DATA_SIZE:") {
           for(int i = 1; i<elems.size(); i++) {
             datasize[i-1] = std::stoi(elems[i]);
             //std::cout << i << " " << datasize[i-1] << std::endl;
           }
        } else if(elems[0] == "DATA_FORMAT:") {
         if(elems[1] == "INT") { 
           dfmt = INT;
         } else if(elems[1] == "FLOAT") {
           dfmt = FLOAT;
         }
        } else if(elems[0] == "VARIABLE:" ) {
           variable = elems[1];
        } else if(elems[0] == "DATA_ENDIAN:") {
           dendian = (elems[1] == "BIG") ? BIG : LITTLE;
        } else if(elems[0] == "DIVIDE_BRICK:") {
         dividebrick = (elems[1] == "true")  ? true : false;
        } else if(elems[0] == "DATA_BRICKLETS:"){
           for(int i = 1; i<elems.size(); i++) bricklets[i-1] = std::stoi(elems[i]);
       }
      }
    }
    myfile.close();
    // index arithmetic, arg...
    if(dividebrick) {
      xpartitions = std::max(datasize[0]/bricklets[0],1);
      ypartitions = std::max(datasize[1]/bricklets[1],1);
      zpartitions = std::max(datasize[2]/bricklets[2],1);
      std::cout << xpartitions << " " << ypartitions << " " << zpartitions << std::endl;
    } else {
      xpartitions = 1;
      ypartitions = 1;
      zpartitions = 1;
    }
    numberofdomains = xpartitions * ypartitions * zpartitions;
  }
  float *readdata(int dom) {
    int domi,domj,domk; // the domain index in global space
    int sample_bytes; // number of bytes in a sample.
    int mydom;
    mydom = dom;
    //unsigned char *samples;
    int *filedata;
    float *samples;
    domi = mydom%xpartitions;
    domj = (mydom/xpartitions)%ypartitions;
    domk = mydom/(xpartitions*ypartitions);
    int istart,jstart,kstart;
    istart = (domi == 0) ? 0 : (domi*bricklets[0] -1);
    jstart = (domj == 0) ? 0 : (domj*bricklets[1] -1);
    kstart = (domk == 0) ? 0 : (domk*bricklets[2] -1);
    //int count[3]; // points in each direction of this domain. 
    counts[0] = (domi == 0) ? bricklets[0] : (bricklets[0] + 1);
    counts[1] = (domj == 0) ? bricklets[1] : (bricklets[1] + 1);
    counts[2] = (domk == 0) ? bricklets[2] : (bricklets[2] + 1);
    origin[0] = (float)istart;
    origin[1] = (float)jstart;
    origin[2] = (float)kstart;
    // read some data. Each domain can only read a single "i" vector at a time
    // make enough space for a single i vector. this should depend on type 
    int *ibuffer = new int[counts[0]];
    // allocate enough space for samples
    samples = new float[counts[0]*counts[1]*counts[2]];
    sample_bytes = sizeof(int);
    char *ptr = (char *)ibuffer;
    std::cout << mydom << " " << domi << " " << domj << " " << domk << std::endl;
    std::cout << mydom << " " << istart << " " << jstart << " " << kstart << std::endl;
    myfile.open(datafile.c_str(), ios::in | ios::binary);
    if(!(myfile.good())) { 
      std::cout << " bad file open " << datafile.c_str() << std::endl;
      exit(1);
    }  
    int dataindex;
    for(int k=kstart;k<kstart+counts[2];k++) 
      for(int j=jstart;j<jstart+counts[1];j++) {
        // read a row of data at a time
        dataindex = k*datasize[0]*datasize[1] + j*datasize[0] + istart;
        streampos src = (k*datasize[0]*datasize[1]+j*datasize[0]+istart)*sample_bytes;
        myfile.seekg(src,ios_base::beg);
        myfile.read(ptr,counts[0]*sample_bytes);
        int offset = counts[0]*((k-kstart)*counts[1] + (j-jstart));
        for(int i=0;i<counts[0];i++) {
          samples[offset+i] = (float)ibuffer[i];
          //std::cout << i<<" " << j << " " << k << " " << dataindex << " " << samples[offset+i]/2.0 << std::endl;
        }
      }
    myfile.close();
    glm::vec3 lower(origin[0],origin[1],origin[2]);
    glm::vec3 upper = lower + glm::vec3((float)counts[0],(float)counts[1],(float)counts[2]) - glm::vec3(1.0,1.0,1.0);
    volbox = new gvt::render::data::primitives::Box3D(lower,upper);
    std::cout << "lower coords " << lower[0] << " " << lower[1] << " " << lower[2] << std::endl;
    std::cout << "upper coords " << upper[0] << " " << upper[1] << " " << upper[2] << std::endl;
    dfmt = FLOAT;

    return samples;
  }
};
// determine if file is a directory
bool isdir(const char *path) {
  struct stat buf;
  stat(path, &buf);
  return S_ISDIR(buf.st_mode);
}
// determine if a file exists
bool file_exists(const char *path) {
  struct stat buf;
  return(stat(path, &buf) == 0);
}
int main(int argc, char **argv) {

  ParseCommandLine cmd("gvtVol");

  cmd.addoption("eye", ParseCommandLine::FLOAT, "Camera position", 3);
  cmd.addoption("look", ParseCommandLine::FLOAT, "Camera look at", 3);
  cmd.addoption("volfile", ParseCommandLine::PATH | ParseCommandLine::REQUIRED, "File path to Volume");
  cmd.addoption("ctffile", ParseCommandLine::PATH | ParseCommandLine::REQUIRED, "File path to color transfer function");
  cmd.addoption("otffile", ParseCommandLine::PATH | ParseCommandLine::REQUIRED, "File path to opacity transfer function");
  cmd.addoption("image", ParseCommandLine::NONE, "schedule", 0);
  cmd.addoption("domain", ParseCommandLine::NONE, "schedule", 0);
  cmd.addoption("wsize", ParseCommandLine::INT, "Window size", 2);
  cmd.addoption("threads", ParseCommandLine::INT, "Number of threads to use (default number cores + ht)", 1);

  cmd.addconflict("image", "domain");

  cmd.parse(argc, argv);

  if (!cmd.isSet("threads")) {
    tbb::task_scheduler_init init(std::thread::hardware_concurrency());
  } else {
    tbb::task_scheduler_init init(cmd.get<int>("threads"));
  }

  MPI_Init(&argc, &argv);
  MPI_Pcontrol(0);
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // context initialization should go in gvt_Init but gvt_Init doesnt exist yet...
  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  if (cntxt == NULL) {
    std::cout << "Something went wrong initializing the context" << std::endl;
    exit(0);
  }
  gvt::core::DBNodeH root = cntxt->getRootNode();
  if (MPI::COMM_WORLD.Get_rank() == 0) {
    cntxt->addToSync(cntxt->createNodeFromType("Data", "Data", root.UUID()));
    cntxt->addToSync(cntxt->createNodeFromType("Instances", "Instances", root.UUID()));
  }
  cntxt->syncContext();

  gvt::core::DBNodeH dataNodes = root["Data"];
  gvt::core::DBNodeH instNodes = root["Instances"];

  std::string filename, filepath, volumefile,otffile,ctffile;
  volumefile = cmd.get<std::string>("volfile");
  ctffile = cmd.get<std::string>("ctffile");
  otffile = cmd.get<std::string>("otffile");
  // volume data...
  if(!file_exists(volumefile.c_str())) {
    cout << "File \"" << volumefile << "\" does not exist. Exiting." << endl;
    return 0;
  }
  if(!file_exists(ctffile.c_str())) {
    cout << "Transfer function file \"" << ctffile << "\" missing. Exiting. " << endl;
    return 0;
  }
  if(!file_exists(otffile.c_str())) {
    cout << "Transfer function file \"" << otffile << "\" missing. Exiting. " << endl;
    return 0;
  }
  
  if(isdir(volumefile.c_str())) {
    cout << "File \"" << volumefile << "\" is a directory. Need a file. Exiting." << endl;
    return 0;
  }
  // read the bov header
  bovheader volheader(volumefile);
  // read the volume .
  // reusing the mesh node type since there is nothing specific to mesh data
  // in the node type. Except the name I suppose
  // typically an mpi rank will only read a subset of the total available domains 
  // provided there are more domains than ranks. If not then each rank will read at most
  // one domain. Either way the particular domain is responsible for creating and 
  // sharing the database node with the rest of the ranks. 
  int worldsize;
  worldsize = MPI::COMM_WORLD.Get_size();
  for(int domain =0; domain < volheader.numberofdomains; domain++) {
    if(domain%worldsize == rank){ // read this domain 
      std::cout << "rank " << rank << " reading domain " << domain <<std::endl;
      gvt::core::DBNodeH VolumeNode = cntxt->addToSync(
        cntxt->createNodeFromType("Mesh",volumefile.c_str(),dataNodes.UUID()));
      // create a volume object which is similar to a mesh object
      gvt::render::data::primitives::Volume *vol = 
        new gvt::render::data::primitives::Volume();
      // read volume file.
      float* sampledata = volheader.readdata(domain);
      gvt::render::data::primitives::TransferFunction *tf = 
        new gvt::render::data::primitives::TransferFunction();
      // read transfer function. 
      tf->load(ctffile,otffile);
      tf->setValueRange(glm::vec2(0.0,65536.0));
      // push the sample data into the volume and fill the other
      // required values in the volume.
      vol->SetVoxelType(gvt::render::data::primitives::Volume::FLOAT);
      vol->SetSamples(sampledata);
      vol->SetTransferFunction(tf);
      vol->SetCounts(volheader.counts[0],volheader.counts[1],volheader.counts[2]);
      vol->SetOrigin(volheader.origin[0],volheader.origin[1],volheader.origin[2]);
      glm::vec3 dels = {1.0,1.0,1.0};
      std::cout << "deltas " << dels.x << " " << dels.y << " " << dels.z << std::endl;
      vol->SetDeltas(dels.x,dels.y,dels.z);
      gvt::render::data::primitives::Box3D *volbox = volheader.volbox;
      // stuff it in the db
      VolumeNode["file"] = volumefile.c_str();
      VolumeNode["bbox"] = (unsigned long long)volbox;
      VolumeNode["ptr"] = (unsigned long long)vol;

      gvt::core::DBNodeH loc = cntxt->createNode("rank",MPI::COMM_WORLD.Get_rank());
      VolumeNode["Locations"] += loc;
      // not sure if I need this call or not based on the way VolumeNode was created.
      cntxt->addToSync(VolumeNode);
    }
  }
    cntxt->syncContext();
      // add instances by looping through the domains again. It is enough for rank 0 to do this. It gets synced in the end. 
  if(MPI::COMM_WORLD.Get_rank()==0) {
    for(int domain=0;domain < volheader.numberofdomains; domain++) {
      gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance", "inst", instNodes.UUID());
      gvt::core::DBNodeH VolumeNode = dataNodes.getChildren()[domain];
      gvt::render::data::primitives::Box3D *mbox = 
        (gvt::render::data::primitives::Box3D *)VolumeNode["bbox"].value().toULongLong();
      instnode["id"] = domain;
      instnode["meshRef"] = VolumeNode.UUID();
      auto m = new glm::mat4(1.f);
      auto minv = new glm::mat4(1.f);
      auto normi = new glm::mat3(1.f);
      instnode["mat"] = (unsigned long long)m;
      *minv = glm::inverse(*m);
      instnode["matInv"] = (unsigned long long)minv;
      *normi = glm::transpose(glm::inverse(glm::mat3(*m)));
      instnode["normi"] = (unsigned long long)normi;
      auto il = glm::vec3((*m) * glm::vec4(mbox->bounds_min, 1.f));
      auto ih = glm::vec3((*m) * glm::vec4(mbox->bounds_max, 1.f));
      gvt::render::data::primitives::Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
      instnode["bbox"] = (unsigned long long)ibox;
      instnode["centroid"] = ibox->centroid();
      cntxt->addToSync(instnode);
    }
  }
  cntxt->syncContext();

  // add lights, camera, and film to the database all nodes do this.
  gvt::core::DBNodeH lightNodes = cntxt->createNodeFromType("Lights", "Lights", root.UUID());
  gvt::core::DBNodeH lightNode = cntxt->createNodeFromType("PointLight", "conelight", lightNodes.UUID());
  lightNode["position"] = glm::vec3(0.0, 0.0, 1.0);
  lightNode["color"] = glm::vec3(100.0, 100.0, 500.0);
  // camera
  gvt::core::DBNodeH camNode = cntxt->createNodeFromType("Camera", "conecam", root.UUID());
  //camNode["eyePoint"] = glm::vec3(640.0, 640.0, 640.0);
  //camNode["eyePoint"] = glm::vec3(127.5,127.5,700.83650);
  camNode["eyePoint"] = glm::vec3(127.5,127.5,1024);
  //camNode["focus"] = glm::vec3(0.0, 0.0, 0.0);
  camNode["focus"] = glm::vec3(127.5, 127.5, 0.0);
  camNode["upVector"] = glm::vec3(0.0,1.0, 0.0);
  //camNode["upVector"] = glm::vec3(-0.42,-0.40, 0.82);
  camNode["fov"] = (float)(30.0 * M_PI / 180.0);
  camNode["rayMaxDepth"] = (int)1;
  camNode["raySamples"] = (int)1;
  // film
  gvt::core::DBNodeH filmNode = cntxt->createNodeFromType("Film", "conefilm", root.UUID());
  filmNode["width"] = 100;
  filmNode["height"] = 100;

  if (cmd.isSet("eye")) {
    std::vector<float> eye = cmd.getValue<float>("eye");
    camNode["eyePoint"] = glm::vec3(eye[0], eye[1], eye[2]);
  }

  if (cmd.isSet("look")) {
    std::vector<float> eye = cmd.getValue<float>("look");
    camNode["focus"] = glm::vec3(eye[0], eye[1], eye[2]);
  }
  if (cmd.isSet("wsize")) {
    std::vector<int> wsize = cmd.getValue<int>("wsize");
    filmNode["width"] = wsize[0];
    filmNode["height"] = wsize[1];
  }

  gvt::core::DBNodeH schedNode = cntxt->createNodeFromType("Schedule", "Enzosched", root.UUID());
  if (cmd.isSet("domain"))
    schedNode["type"] = gvt::render::scheduler::Domain;
  else
    schedNode["type"] = gvt::render::scheduler::Image;

#ifdef GVT_RENDER_ADAPTER_EMBREE
  int adapterType = gvt::render::adapter::Embree;
#elif GVT_RENDER_ADAPTER_MANTA
  int adapterType = gvt::render::adapter::Manta;
#elif GVT_RENDER_ADAPTER_OPTIX
  int adapterType = gvt::render::adapter::Optix;
#elif GVT_RENDER_ADAPTER_OSPRAY
  int adapterType = gvt::render::adapter::Ospray;
#elif
  GVT_DEBUG(DBG_ALWAYS, "ERROR: missing valid adapter");
#endif

  //schedNode["adapter"] = gvt::render::adapter::Embree;
  schedNode["adapter"] = adapterType;

  // end db setup
  //
  //cntxt->database()->printTree(root.UUID(),1,0,std::cout);

  // use db to create structs needed by system

  // setup gvtCamera from database entries
  gvt::render::data::scene::gvtPerspectiveCamera mycamera;
  glm::vec3 cameraposition = camNode["eyePoint"].value().tovec3();
  glm::vec3 focus = camNode["focus"].value().tovec3();
  float fov = camNode["fov"].value().toFloat();
  glm::vec3 up = camNode["upVector"].value().tovec3();
  int rayMaxDepth = camNode["rayMaxDepth"].value().toInteger();
  int raySamples = camNode["raySamples"].value().toInteger();
  mycamera.lookAt(cameraposition, focus, up);
  mycamera.setMaxDepth(rayMaxDepth);
  mycamera.setSamples(raySamples);
  mycamera.setFOV(fov);
  mycamera.setFilmsize(filmNode["width"].value().toInteger(), filmNode["height"].value().toInteger());

#ifdef GVT_USE_MPE
  MPE_Log_event(readend, 0, NULL);
#endif
  // setup image from database sizes
  gvt::render::data::scene::Image myimage(mycamera.getFilmSizeWidth(), mycamera.getFilmSizeHeight(), "enzo");

  mycamera.AllocateCameraRays();
  mycamera.generateRays();

  int schedType = root["Schedule"]["type"].value().toInteger();
  switch (schedType) {
  case gvt::render::scheduler::Image: {
    std::cout << "starting image scheduler" << std::endl;
#ifdef GVT_RENDER_ADAPTER_OSPRAY
    //gvt::render::algorithm::Tracer<gvt::render::schedule::ImageScheduler> tracer(&argc,argv,mycamera.rays, myimage);
#endif
    gvt::render::algorithm::Tracer<gvt::render::schedule::ImageScheduler> tracer(mycamera.rays, myimage);
    for (int z = 0; z < 10; z++) {
      mycamera.AllocateCameraRays();
      mycamera.generateRays();
      myimage.clear();
      tracer();
    }
    break;
  }
  case gvt::render::scheduler::Domain: {
    std::cout << "starting domain scheduler" << std::endl;
#ifdef GVT_USE_MPE
    MPE_Log_event(renderstart, 0, NULL);
#endif
    // gvt::render::algorithm::Tracer<DomainScheduler>(mycamera.rays, myimage)();
#ifdef GVT_RENDER_ADAPTER_OSPRAY
    gvt::render::algorithm::Tracer<gvt::render::schedule::DomainScheduler> tracer(&argc,argv,mycamera.rays, myimage);
#elif
    gvt::render::algorithm::Tracer<gvt::render::schedule::DomainScheduler> tracer(mycamera.rays, myimage);
#endif
    for (int z = 0; z < 1; z++) {
      mycamera.AllocateCameraRays();
      mycamera.generateRays();
      if(MPI::COMM_WORLD.Get_rank()==0) 
     //   mycamera.dumpraystostdout();
      myimage.clear();
      tracer();
    }
    break;
#ifdef GVT_USE_MPE
    MPE_Log_event(renderend, 0, NULL);
#endif
    break;
  }
  default: {
    std::cout << "unknown schedule type provided: " << schedType << std::endl;
    break;
  }
  }

  myimage.Write();
#ifdef GVT_USE_MPE
  MPE_Log_sync_clocks();
// MPE_Finish_log("gvtSimplelog");
#endif
  if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();
}

