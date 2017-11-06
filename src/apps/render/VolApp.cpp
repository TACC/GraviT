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
 * the command line to run this application from a subdirectory of the GraviT
 * root directory is 
 *
 * ibrun bin/gvtVol -volfile ../data/vol/int8.bov -domain -ctffile ../data/colormaps/Grayscale.cmap -otffile ../data/colormaps/Grayscale.omap -wsize 512,512
 *
 *
 *
*/
#include <algorithm>
#include <gvt/core/Math.h>
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
#define USEAPI
#ifdef USEAPI
#include <gvt/render/api/api.h>
#endif

using namespace std;

// split a string into parts using the delimiter given return 
// the parts in a vector of strings. 
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
    SHORT,
    UNKNOWN
  } dfmt,odfmt,idfmt;
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
  // read the header file to determine the metadata
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
           }
        } else if(elems[0] == "DATA_FORMAT:") {
         if(elems[1] == "INT") { 
           idfmt = INT;
         } else if(elems[1] == "FLOAT") {
           idfmt = FLOAT;
         } else if(elems[1] == "UINT") {
           idfmt = UINT;
         } else if(elems[1] == "SHORT") {
           idfmt = SHORT;
         } else {
           cout << " BovReader: Unrecognized datatype " << endl;
           idfmt = UNKNOWN;
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
    } else {
      xpartitions = 1;
      ypartitions = 1;
      zpartitions = 1;
    }
    numberofdomains = xpartitions * ypartitions * zpartitions;
  }
  // read the scalar field itself
  float *readdata(int dom) {
    int domi,domj,domk; // the domain index in global space
    int sample_bytes; // number of bytes in a sample.
    int mydom;
    mydom = dom;
    int *filedata;
    float *samples;
    domi = mydom%xpartitions;
    domj = (mydom/xpartitions)%ypartitions;
    domk = mydom/(xpartitions*ypartitions);
    int istart,jstart,kstart;
    istart = (domi == 0) ? 0 : (domi*bricklets[0] -1);
    jstart = (domj == 0) ? 0 : (domj*bricklets[1] -1);
    kstart = (domk == 0) ? 0 : (domk*bricklets[2] -1);
    counts[0] = (domi == 0) ? bricklets[0] : (bricklets[0] + 1);
    counts[1] = (domj == 0) ? bricklets[1] : (bricklets[1] + 1);
    counts[2] = (domk == 0) ? bricklets[2] : (bricklets[2] + 1);
    origin[0] = (float)istart;
    origin[1] = (float)jstart;
    origin[2] = (float)kstart;
    std::cout << istart << " " << jstart << " " <<kstart << std::endl;
    std::cout << counts[0] << " " << counts[1] << " " << counts[2] << std::endl;
    myfile.open(datafile.c_str(), ios::in | ios::binary);
    if(!(myfile.good())) { 
      std::cout << " bad file open " << datafile.c_str() << std::endl;
      exit(1);
    }  
    // read some data. Each domain can only read a single "i" vector at a time
    // make enough space for a single i vector. this should depend on type 
    // allocate enough space for samples
    samples = new float[counts[0]*counts[1]*counts[2]];
    switch (idfmt) {
      case INT : {
        sample_bytes = sizeof(int);
        int *ibuffer = new int[counts[0]];
        char *ptr = (char *)ibuffer;
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
            }
          }
        std::cout << "data " << samples[0] << std::endl;
        break;
      }
      case FLOAT : {
        sample_bytes = sizeof(float);
        int *ibuffer = new int[counts[0]];
        float *fbuffer = new float[counts[0]];
        char *ptr = (char *)fbuffer;
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
          samples[offset+i] = fbuffer[i];
        }
      }
          break;
        }
      default: {
                 std::cout << " messed up type " << idfmt << std::endl;
                 break;
               }
    }
    myfile.close();
    glm::vec3 lower(origin[0],origin[1],origin[2]);
    glm::vec3 upper = lower + glm::vec3((float)counts[0],(float)counts[1],(float)counts[2]) - glm::vec3(1.0,1.0,1.0);
    volbox = new gvt::render::data::primitives::Box3D(lower,upper);
    odfmt = FLOAT;
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

  cmd.addoption("upVector", ParseCommandLine::FLOAT, "upVector", 3);
  cmd.addoption("eye", ParseCommandLine::FLOAT, "Camera position", 3);
  cmd.addoption("look", ParseCommandLine::FLOAT, "Camera look at", 3);
  cmd.addoption("volfile", ParseCommandLine::PATH | ParseCommandLine::REQUIRED, "File path to Volume");
  cmd.addoption("imagefile", ParseCommandLine::PATH , "image file name");
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

#ifdef USEAPI
  // API initialization
  gvtInit(argc,argv);
  int rank,worldsize;
  // get rank and world size for use downstream
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
#else
  // this preprocessor block does what gvtInit does
  MPI_Init(&argc, &argv);
  MPI_Pcontrol(0);
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int worldsize;
  worldsize = MPI::COMM_WORLD.Get_size();

  // context initialization should go in gvt_Init but gvt_Init doesnt exist yet...
  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();

  gvt::core::DBNodeH root = cntxt->getRootNode();
  root += cntxt->createNode(
          "threads", cmd.isSet("threads") ? (int)cmd.get<int>("threads") : (int)std::thread::hardware_concurrency());


  if (cntxt == NULL) {
    std::cout << "Something went wrong initializing the context" << std::endl;
    exit(0);
  }
//  gvt::core::DBNodeH root = cntxt->getRootNode();
  if (MPI::COMM_WORLD.Get_rank() == 0) {
    cntxt->addToSync(cntxt->createNodeFromType("Data", "Data", root.UUID()));
    cntxt->addToSync(cntxt->createNodeFromType("Instances", "Instances", root.UUID()));
  }
  cntxt->syncContext();
  gvt::core::DBNodeH dataNodes = root["Data"];
  gvt::core::DBNodeH instNodes = root["Instances"];
#endif


  // read volume information and initialize gravit volume object
  // transfer functions are associated with the volume
  std::string filename, filepath, volumefile,otffile,ctffile,imagefile,volnodename;
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
  for(int domain =0; domain < volheader.numberofdomains; domain++) {
    if(domain%worldsize == rank){ // read this domain 
        std::cout << " rank " << rank << " reading domain " << domain << std::endl;
      // create a volume object which is similar to a mesh object
      gvt::render::data::primitives::Volume *vol = 
        new gvt::render::data::primitives::Volume();
      // read volume file.
      float* sampledata = volheader.readdata(domain);
      gvt::render::data::primitives::TransferFunction *tf = 
        new gvt::render::data::primitives::TransferFunction();
      // read transfer function. 
      tf->load(ctffile,otffile);
      // this value range is for small enzo data
      tf->setValueRange(glm::vec2(0.0,65536.0));
      // required values in the volume.
      vol->SetVoxelType(gvt::render::data::primitives::Volume::FLOAT);
      vol->SetSamples(sampledata);
      vol->SetTransferFunction(tf);
      vol->SetCounts(volheader.counts[0],volheader.counts[1],volheader.counts[2]);
      vol->SetOrigin(volheader.origin[0],volheader.origin[1],volheader.origin[2]);
      float deltas[3] = {1.0,1.0,1.0};
      float samplingrate = 1.0;
      glm::vec3 dels = {1.0,1.0,1.0};
      vol->SetDeltas(dels.x,dels.y,dels.z);
      vol->SetSamplingRate(samplingrate);
      gvt::render::data::primitives::Box3D *volbox = volheader.volbox;
      // stuff it in the db
#ifdef USEAPI
      // create a mesh object, add it to the db 
      // but we need a unique name for each actual mesh. 
      // for now add the domain number to the volumefile name. 
      // It will work.. trust me... 
      std::cout << "create volume and add samples " << volnodename << std::endl;
      volnodename = volumefile + std::to_string(domain);
      createVolume(volnodename);
      addVolumeTransferFunctions(volnodename,ctffile,otffile,0.0,65536.0);
      addVolumeSamples(volnodename,sampledata,volheader.counts,volheader.origin,deltas,samplingrate);

    }
  }
#else
      gvt::core::DBNodeH VolumeNode = cntxt->addToSync(
        cntxt->createNodeFromType("Mesh",volumefile.c_str(),dataNodes.UUID()));
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
#endif
      // add instances by looping through the domains again. It is enough for rank 0 to do this. It gets synced in the end. 
  if(MPI::COMM_WORLD.Get_rank()==0) {
    for(int domain=0;domain < volheader.numberofdomains; domain++) {
#ifdef USEAPI
      volnodename = volumefile + std::to_string(domain);
      auto m = new glm::mat4(1.f);
      auto &mi = (*m);

              float mf[] = { mi[0][0], mi[0][1], mi[0][2], mi[0][3], mi[1][0], mi[1][1], mi[1][2], mi[1][3],
                                         mi[2][0], mi[2][1], mi[2][2], mi[2][3], mi[3][0], mi[3][1], mi[3][2], mi[3][3] };
      addInstance(volnodename, mf);
    }
  }
#else
      gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance", "inst", instNodes.UUID());
      gvt::core::DBNodeH VolumeNode = dataNodes.getChildren()[domain];
      gvt::render::data::primitives::Box3D *mbox = 
        (gvt::render::data::primitives::Box3D *)VolumeNode["bbox"].value().toULongLong();
      instnode["id"] = domain;
      instnode["meshRef"] = VolumeNode.UUID();
      // we dont transform or use instancing in this example 
      // We load the identity matrix transformations anyway. 
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
#endif

  // add lights, camera, and film to the database all nodes do this.
  // again some default stuff loaded in. Not entirely required in this
  // instance but get in tha habbit of putting it there anyway.
#ifdef USEAPI
  // not sure I need a light but what the heck. 
  auto lpos = glm::vec3(0.,0.,1.);
  auto lcolor = glm::vec3(100.,100.,500.);
  string lightname = "mylight";
  addPointLight(lightname,glm::value_ptr(lpos),glm::value_ptr(lcolor));
  // camera time
  auto eye = glm::vec3(127.5,127.5,1024.);
  if (cmd.isSet("eye")) {
    std::vector<float> cameye = cmd.getValue<float>("eye");
    eye = glm::vec3(cameye[0], cameye[1], cameye[2]);
  }
  auto focus = glm::vec3(127.5,127.5,0.0);
  if (cmd.isSet("look")) {
    std::vector<float> foc = cmd.getValue<float>("look");
    focus = glm::vec3(foc[0], foc[1], foc[2]);
  }
  auto upVector = glm::vec3(0.,1.,0.);
  if (cmd.isSet("upVector")) {
    std::vector<float> upvec = cmd.getValue<float>("upVector");
    upVector = glm::vec3(upvec[0], upvec[1], upvec[2]);
  }
  float fov = (float)(30.0*M_PI/180.0);
  int rayMaxDepth = (int)1;
  int raySamples = (int)1;
  float jitterWindowSize = (float)0.5;
  string camname = "conecam";
  std::cout << "add camera " << camname << std::endl;
  addCamera(camname,glm::value_ptr(eye),glm::value_ptr(focus),glm::value_ptr(upVector),fov,rayMaxDepth,raySamples,jitterWindowSize);
  // film
  string filmname = "conefilm";
  std::cout << "add film " << filmname << std::endl;
  int width = 100;
  int height = 100;
  if (cmd.isSet("wsize")) {
    std::vector<int> wsize = cmd.getValue<int>("wsize");
    width = wsize[0];
    height = wsize[1];
  }
  string outputpath = "volapptest";
  if(cmd.isSet("imagefile")) {
    outputpath = cmd.get<std::string>("imagefile");
  } 
  addFilm(filmname,width,height,outputpath);
  // render bits (schedule and adapter)
  string rendername("VolumeRenderer");
  int schedtype;
  int adaptertype;
  // right now only the domain schedule works for volume rendering
  schedtype = gvt::render::scheduler::Domain;
  // and it only works with the ospray adapter.
#ifdef GVT_RENDER_ADAPTER_OSPRAY
  adaptertype = gvt::render::adapter::Ospray;
#elif
  GVT_DEBUG(DBG_ALWAYS, "ERROR: missing valid adapter");
#endif
  std::cout << "add renderer " << rendername << " " << adaptertype << " " << schedtype << std::endl;
  addRenderer(rendername,adaptertype,schedtype);
  render(rendername);
  writeimage(rendername);
  if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();
#else
  gvt::core::DBNodeH lightNodes = cntxt->createNodeFromType("Lights", "Lights", root.UUID());
  gvt::core::DBNodeH lightNode = cntxt->createNodeFromType("PointLight", "conelight", lightNodes.UUID());
  lightNode["position"] = glm::vec3(0.0, 0.0, 1.0);
  lightNode["color"] = glm::vec3(100.0, 100.0, 500.0);
  // camera. this stuff is required. No camera no image.
  gvt::core::DBNodeH camNode = cntxt->createNodeFromType("Camera", "conecam", root.UUID());
  camNode["eyePoint"] = glm::vec3(127.5,127.5,1024);
  camNode["focus"] = glm::vec3(127.5, 127.5, 0.0);
  camNode["upVector"] = glm::vec3(0.0,1.0, 0.0);
  camNode["fov"] = (float)(30.0 * M_PI / 180.0);
  camNode["rayMaxDepth"] = (int)1;
  camNode["raySamples"] = (int)1;
  // film
  gvt::core::DBNodeH filmNode = cntxt->createNodeFromType("Film", "conefilm", root.UUID());
  filmNode["width"] = 100;
  filmNode["height"] = 100;

  // override some defaults...
  if (cmd.isSet("eye")) {
    std::vector<float> eye = cmd.getValue<float>("eye");
    camNode["eyePoint"] = glm::vec3(eye[0], eye[1], eye[2]);
  }

  if (cmd.isSet("upVector")) {
    std::vector<float> upvec = cmd.getValue<float>("upVector");
    camNode["upVector"] = glm::vec3(upvec[0], upvec[1], upvec[2]);
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

  // at this point application only works with ospray adapter
#ifdef GVT_RENDER_ADAPTER_OSPRAY
  int adapterType = gvt::render::adapter::Ospray;
#elif
  GVT_DEBUG(DBG_ALWAYS, "ERROR: missing valid adapter");
#endif

  schedNode["adapter"] = adapterType;

  // end db setup
  //
  // use db to create structs needed by system

#if 0
  if(rank == 0) 
  cntxt->database()->printTree(root.UUID(),3,std::cout);
#endif
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

  // setup image from database sizes
  if(cmd.isSet("imagefile")) {
    imagefile = cmd.get<std::string>("imagefile");
  } else {
    imagefile = string("volapptest");
  }
  gvt::render::data::scene::Image myimage(mycamera.getFilmSizeWidth(), mycamera.getFilmSizeHeight(), imagefile.c_str());

  mycamera.AllocateCameraRays();
  mycamera.generateRays(true);
  //std::cout << " Initial camera rays " << std::endl;
  //mycamera.dumpraystostdout();

  int schedType = root["Schedule"]["type"].value().toInteger();
  switch (schedType) {
  case gvt::render::scheduler::Image: {
#ifdef GVT_RENDER_ADAPTER_OSPRAY
    // todo fix image scheduler to use ospray
    //gvt::render::algorithm::Tracer<gvt::render::schedule::ImageScheduler> tracer(&argc,argv,mycamera.rays, myimage);
#endif
    gvt::render::algorithm::Tracer<gvt::render::schedule::ImageScheduler> tracer(mycamera.rays, myimage);
    for (int z = 0; z < 10; z++) {
      mycamera.AllocateCameraRays();
      mycamera.generateRays(true);
      myimage.clear();
      tracer();
    }
    break;
  }
  case gvt::render::scheduler::Domain: {
#ifdef GVT_RENDER_ADAPTER_OSPRAY
    gvt::render::algorithm::Tracer<gvt::render::schedule::DomainScheduler> tracer(&argc,argv,mycamera.rays, myimage);
#elif
    gvt::render::algorithm::Tracer<gvt::render::schedule::DomainScheduler> tracer(mycamera.rays, myimage);
#endif
    for (int z = 0; z < 1; z++) {
      mycamera.AllocateCameraRays();
      mycamera.generateRays(true);
      myimage.clear();
      tracer();
    }
    break;
    //break;
  }
  default: {
    std::cout << "unknown schedule type provided: " << schedType << std::endl;
    break;
  }
  }

  myimage.Write();
  if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();
#endif
}

