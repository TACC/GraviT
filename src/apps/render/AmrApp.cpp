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
}
