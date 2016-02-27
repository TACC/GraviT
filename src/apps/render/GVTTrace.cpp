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
//
//  GVTTrace.C
//

#if 0
#include "ConfigFileLoader.h"
#include "MantaRayTracer.h"
#include "OptixRayTracer.h"
#include "EmbreeRayTracer.h"

#include <gvt/core/Math.h>
#ifdef GVT_RENDER_ADAPTER_MANTA
#include <gvt/render/adapter/manta/Wrapper.h>
#endif
#ifdef GVT_RENDER_ADAPTER_OPTIX
#include <gvt/render/adapter/optix/Wrapper.h>
#endif
#ifdef GVT_RENDER_ADAPTER_EMBREE
#include <gvt/render/adapter/embree/Wrapper.h>
#endif
#include <gvt/render/data/Primitives.h>

#include <mpi.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace gvtapps::render;

/// command-line renderer using compile-time specified engine
/**
  command-line ray tracing renderer that uses the engine(s) activated
  during CMake configuration. Available engines include:
   - Manta (mantawiki.sci.utah.edu)
   - Embree (embree.github.io)
   - Optix Prime (developer.nvidia.com/optix)
 */
int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);
  MPI_Barrier(MPI_COMM_WORLD);

  string filename, imagename;

  if (argc > 1)
    filename = argv[1];
  else
    filename = "./gvttrace.conf";

  if (argc > 2)
    imagename = argv[2];
  else
    imagename = "GVTTrace";

  gvtapps::render::ConfigFileLoader cl(filename);

  bool domain_choosen = false;
#ifdef GVT_RENDER_ADAPTER_MANTA
  GVT_DEBUG(DBG_ALWAYS, "Rendering with Manta");
  if (cl.domain_type == 0) {
    domain_choosen = true;
    MantaRayTracer rt(cl);
    MPI_Barrier(MPI_COMM_WORLD);
    rt.RenderImage(imagename + "_manta");
  }
#endif
#ifdef GVT_RENDER_ADAPTER_OPTIX
  GVT_DEBUG(DBG_ALWAYS, "Rendering with OptiX");
  if (cl.domain_type == 1) {
    domain_choosen = true;
    OptixRayTracer rt(cl);
    MPI_Barrier(MPI_COMM_WORLD);
    rt.RenderImage(imagename + "_optix");
  }
#endif
#ifdef GVT_RENDER_ADAPTER_EMBREE
  GVT_DEBUG(DBG_ALWAYS, "Rendering with Embree");
  if (cl.domain_type == 2) {
    domain_choosen = true;
    EmbreeRayTracer rt(cl);
    MPI_Barrier(MPI_COMM_WORLD);
    rt.RenderImage(imagename + "_embree");
  }
#endif

  GVT_ASSERT(domain_choosen, "The requested domain type is not available, please recompile");

  if (MPI::COMM_WORLD.Get_size() > 1)
    MPI_Finalize();

  return 0;
}
#endif