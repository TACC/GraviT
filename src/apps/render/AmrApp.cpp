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
 * This application tests the Galaxy AMR volume rendering adapter. 
 *
 * Reads the amrvol datatype from Galaxy and renders the volume. 
 *
 * The command line torun this application from a subdirectory of the GraviT 
 * root directory:
 *
 * bin/gvtAmr -volfile ../data/vol/
 *
 */
#include <algorithm>
#include <gvt/core/Math.h>

#include <gvt/render/api/api.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/Types.h>
#include <gvt/render/data/Domains.h>
#include <set>
#include <vector>

#include <tbb/task_scheduler_init.h>
#include <thread>

#if defined GVT_RENDER_ADAPTER_GALAXY
#include <gvt/render/adapter/galaxy/PVolAdapter.h>
#else
#error "Must define Galaxy adapter"
#endif

#include <vtkStructuredPointsReader.h>
#include <vtkStructuredPoints.h>
#include <vtkSmartPointer.h>

#include <iostream>
#include <fstream>
#include <ios>
#include <math.h>
#include <glob.h>
#include <sys/stat.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <queue>

#include "ParseCommandLine.h"

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

// amrvol reader stuff
// This bit imports the amrvol dataset. amrvol uses the vtk datasets to hold
// grids so we need to use some vtk.
struct amrheader {
    std::string amrvolfile;
    int numberoflevels;
    int numberofgrids;
    std::vector<int> gridsperlevel;
    std::vector<std::string> gridfilenames;
    std::vector<std::vector<int>> subgrids; 
int getnumberoflevels() { return numberoflevels;}
int getgridsinlevel(int level) { return gridsperlevel[level];}
std::string getgridfilename(int grid) { return gridfilenames[grid];}

amrheader(std::string filename): amrvolfile(filename) {
    ifstream in;
    int parentgridindex;
    in.open(amrvolfile.c_str());
    if(in.fail())
    {
        std::cerr << "ERROR: unable to open volfile:" << amrvolfile << std::endl;
        exit(1);
    }
    in >> numberoflevels; // get number of levels
    int grids;
    std::string gfn;
    numberofgrids = 0;
    for(int level = 0;level < numberoflevels; level++)
    {
        in>>grids;
        gridsperlevel.push_back(grids);
        numberofgrids += gridsperlevel[level];
    }
    // this loop reads the grid file names and parent grid index
    // if the parent grid index of the grid being read is >=0
    // the grid is placed on the subgrid vector for that grid.
    // For example, if i = 5 (the sixth grid in the collection)
    // and the parentgridindex is 0 then this grid (5) is a
    // subgrid of the parentgridindex (0). So we push this grid
    // index (5) onto the vector of subgrids for grid parentgridindex (0).
    // it remains to be seen if I actually make use of this information..
    for(int i = 0;i<numberofgrids;i++)
    {
        in>>gfn,parentgridindex;
        // if we nave a nonzero parent index stash this grid index 
        // in the subgrid vector of the parent. If not create an
        // empty subgrid vector for this index because the index
        // is a level 0 grid index. The input file needs all the 
        // level 0 grids to be listed first.
        if(!(parentgridindex < 0)) 
            subgrids[parentgridindex].push_back(i);
        else
            subgrids.push_back(std::vector<int>());
        gridfilenames.push_back(gfn);
    }
}

};

// this thing grabs a copy of the sample data from the vtk
// object. Memory for the data is dynamically allocated 
// so it does not go away when the vtk object gets deleted. 
float * loadsamplesfromvtk(vtkStructuredPoints * gsp) {
    int npts = gsp->GetNumberOfPoints();
    float * vtksamples =(float*) gsp->GetScalarPointer();
    float * fltpts = new float[npts];
    for(int i=0;i<npts;i++)
        fltpts[i] = vtksamples[i];
    return fltpts;
}
int main(int argc, char ** argv) {

  ParseCommandLine cmd("gvtAmr");

  cmd.addoption("upVector", ParseCommandLine::FLOAT, "upVector", 3);
  cmd.addoption("eye", ParseCommandLine::FLOAT, "Camera position", 3);
  cmd.addoption("look", ParseCommandLine::FLOAT, "Camera look at", 3);
  cmd.addoption("volfile", ParseCommandLine::PATH , "File path to Volume");
  cmd.addoption("imagefile", ParseCommandLine::PATH , "image file name");
  cmd.addoption("ctffile", ParseCommandLine::PATH , "File path to color transfer function");
  cmd.addoption("otffile", ParseCommandLine::PATH , "File path to opacity transfer function");
  cmd.addoption("domain", ParseCommandLine::NONE, "schedule", 0);
  cmd.addoption("adomain", ParseCommandLine::NONE, "schedule", 0);
  cmd.addoption("wsize", ParseCommandLine::INT, "Window size", 2);
  cmd.addoption("threads", ParseCommandLine::INT, "Number of threads to use (default number cores + ht)", 1);

  cmd.addconflict("image", "domain");

  cmd.parse(argc, argv);

  if (!cmd.isSet("threads")) { // use all hardware threads
    tbb::task_scheduler_init init(std::thread::hardware_concurrency());
    api::gvtInit(argc,argv);
  } else {                    // use specific number of threads.
    tbb::task_scheduler_init init(cmd.get<int>("threads"));
    api::gvtInit(argc,argv,cmd.get<int>("threads"));
  }

  // get mpi bits..
  int rank,worldsize;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&worldsize);

  // read volume information and initialize gravit volume object.
  std::string filename, filepath, volumefile,otffile,ctffile,imagefile,volnodename;
  volumefile = "../data/vol/ballinthecorner.amrvol";
  ctffile = "../data/colormaps/Grayscale.cmap";
  otffile = "../data/colormaps/Grayscale.omap";
  // transfer functions
  if(cmd.isSet("volfile"))
    volumefile = cmd.get<std::string>("volfile");
  if(cmd.isSet("ctffile")) {
    ctffile = cmd.get<std::string>("ctffile");
    std::cerr << " using " << ctffile << " for color " << std::endl;
  }
  if(cmd.isSet("otffile"))
    otffile = cmd.get<std::string>("otffile");
  // volume data
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
  // open the file and read the data
  amrheader amrmetadata(volumefile);
  int levels = amrmetadata.numberoflevels;
  int gridsinlevel;
  int gridindex = 0;
  std::string gridfilename;
  std::cerr << levels<< std::endl;
  for(int lev=0;lev<levels;lev++) {
      gridsinlevel = amrmetadata.gridsperlevel[lev];
      std::cerr << "level " << lev << " has " << gridsinlevel << " grids"<< std::endl ;
      for(int grd=0; grd<gridsinlevel; grd++) {
          gridfilename = amrmetadata.gridfilenames[gridindex];
          std::cerr << "\t grid " << grd << " level " << lev << " is in file " << gridfilename<< std::endl;
          gridindex++;
      }
  }
  std::cerr << "total number of grids " << amrmetadata.numberofgrids << std::endl;
  // now read the domains (grids in level 0 and subgrids)
  int numberofdomains;
  // going to need some vtk in here...
  vtkSmartPointer<vtkStructuredPointsReader> gridreader = vtkSmartPointer<vtkStructuredPointsReader>::New();
  vtkStructuredPoints *gsp;
  double dvector[3],range[2],bounds[6];
  int ivector[3];
  float origin[3],spacing[3];
  float *fltptr;
  float *vtkfloats;
  bool volisamr = true;
  std::string dir((volumefile.find_last_of("/") == std::string::npos) ? "" :
                      volumefile.substr(0,volumefile.find_last_of("/")+1));
  numberofdomains = amrmetadata.gridsperlevel[0];
  float samplingrate = 1.0; // bad idea to hardwire but well..
  for(int domain = 0;domain< numberofdomains; domain++) {
      if(domain%worldsize == rank) { // read this domain and load some local struct
          std::cout << " rank " << rank << " reading domain " << domain << std::endl;
          //read the level 0 grid information
          gridfilename = dir + amrmetadata.gridfilenames[domain];
          gridreader->SetFileName(gridfilename.c_str());
          gsp = gridreader->GetOutput();
          gridreader->Update();
          // pull metadata for this grid
          volnodename = amrmetadata.gridfilenames[domain] + std::to_string(domain);
          api::createVolume(volnodename,volisamr);
          gsp->GetOrigin(dvector);
          origin[0] = dvector[0];
          origin[1] = dvector[1];
          origin[2] = dvector[2];
          gsp->GetSpacing(dvector);
          spacing[0] = dvector[0];
          spacing[1] = dvector[1];
          spacing[2] = dvector[2];
          gsp->ComputeBounds();
          gsp->GetDimensions(ivector);
          gsp->GetBounds(bounds);
          //Deep copy the sample data from the vtk object
          fltptr = loadsamplesfromvtk(gsp);
          // store the level 0 grid info in the volume
          api::addVolumeSamples(volnodename,fltptr,ivector,origin,spacing,samplingrate);
          // find the grids that are contained in this level 0 grid
          // I have a vector of subgrids which contains the indices
          // of the subgrids of this level0 grid. I need to collect the
          // subgrids indices and then decend the tree to collect the
          // subrrid indices of each subindex. Its a tree search. 
          std::vector<int> domainsubgrids;
          // initialize with level 1 grids
          std::queue<int> searchqueue; 
          for(int i : amrmetadata.subgrids[domain])
              searchqueue.push(i);
          // search for more
          int sgrid;
          while(!searchqueue.empty()) {
              // add element to domain subgrids
              sgrid = searchqueue.front();
              domainsubgrids.push_back(sgrid);
              searchqueue.pop();
              for(int j : amrmetadata.subgrids[sgrid])
                  searchqueue.push(j);
          }
          // domainsubgrids should contain all subgrids of this domain
          // parse them and load their data via the api
          for(int k : domainsubgrids) {
              // read from vtk file
            gridfilename = dir + amrmetadata.gridfilenames[k];
            gridreader->SetFileName(gridfilename.c_str());
            gsp = gridreader->GetOutput();
            gridreader->Update();
            gsp->GetOrigin(dvector);
            origin[0] = dvector[0];
            origin[1] = dvector[1];
            origin[2] = dvector[2];
            gsp->GetSpacing(dvector);
            spacing[0] = dvector[0];
            spacing[1] = dvector[1];
            spacing[2] = dvector[2];
            gsp->ComputeBounds();
            gsp->GetDimensions(ivector);
            fltptr = loadsamplesfromvtk(gsp);
            api::addAmrSubgrid(volnodename,domainsubgrids[k],fltptr,ivector,origin,spacing);
          } // loop over subgrids
      } // if level 0 grid is in this rank
  } // loop over level 0 grids (domains)
  api::gvtsync();
  // now for instancing. 
  if(rank == 0) {
      for(int domain = 0;domain <numberofdomains;domain++) {
          volnodename = amrmetadata.gridfilenames[domain] + std::to_string(domain);
          auto m = new glm::mat4(1.f);
          auto &mi = (*m);
          float mf[] = {mi[0][0], mi[0][1], mi[0][2], mi[0][3], mi[1][0], mi[1][1], mi[1][2], mi[1][3],
              mi[2][0], mi[2][1], mi[2][2], mi[2][3], mi[3][0], mi[3][1], mi[3][2], mi[3][3] };
          api::addInstance(std::string("inst")+std::to_string(domain),volnodename, mf);
      }
  }
  // lights etc goes here...
  auto eye = glm::vec3(3.,3.,3.);
  if(cmd.isSet("eye")) {
      std::vector<float> cameye = cmd.getValue<float>("eye");
      eye = glm::vec3(cameye[0],cameye[1],cameye[2]);
  }
  auto focus = glm::vec3(-4.0,-4.0,-4.0);
  auto upVector = glm::vec3(0.0,0.0,1.0);
  auto fov = (float)(30.0*M_PI/180.0);
  int rayMaxDepth = (int)1;
  int raySamples = (int)1;
  float jitterWindowSize = (float)0.5;
  string camname = "ballcam";
  api::addCamera(camname,glm::value_ptr(eye),glm::value_ptr(focus),glm::value_ptr(upVector),fov,rayMaxDepth,raySamples,jitterWindowSize);
  //film
  string filmname = "ballfilm";
  int width = 10;
  int height = 10;
  if(cmd.isSet("wsize")) {
      std::vector<int> wsize = cmd.getValue<int>("wsize");
      width = wsize[0];
      height = wsize[1];
  }
  string outputpath = "ballapptest";
  if(cmd.isSet("imagefile")) {
      outputpath = cmd.get<std::string>("imagefile");
  }
  api::addFilm(filmname,width,height,outputpath);
  string rendername("VolumeRenderer");
  int schedtype;
  int adaptertype;
  if(cmd.isSet("adomain"))
      schedtype = gvt::render::scheduler::AsyncDomain;
  else
      schedtype = gvt::render::scheduler::Domain;
#ifdef GVT_RENDER_ADAPTER_GALAXY
  adaptertype = gvt::render::adapter::Pvol;
#else
  GVT_DEBUG(DBG_ALWAYS, "ERROR: missing valid adapter");
#endif
  api::gvtsync();
  api::addRenderer(rendername,adaptertype,schedtype,camname,filmname,true);
  api::gvtsync();
  api::render(rendername);
  if(worldsize > 1) MPI_Finalize();
}
