//
//  GVTTrace.C
//

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

int main(int argc, char** argv) {

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
  GVT_DEBUG(DBG_ALWAYS,"Rendering with Manta");
  if (cl.domain_type == 0) {
    domain_choosen = true;
    MantaRayTracer rt(cl);
    MPI_Barrier(MPI_COMM_WORLD);
    rt.RenderImage(imagename);
  }
#endif
#ifdef GVT_RENDER_ADAPTER_OPTIX
  GVT_DEBUG(DBG_ALWAYS,"Rendering with OptiX");
  if (cl.domain_type == 1) {
    domain_choosen = true;
    OptixRayTracer rt(cl);
    MPI_Barrier(MPI_COMM_WORLD);
    rt.RenderImage(imagename);
  }
#endif
#ifdef GVT_RENDER_ADAPTER_EMBREE
  GVT_DEBUG(DBG_ALWAYS,"Rendering with Embree");
  if (cl.domain_type == 2) {
    domain_choosen = true;
    EmbreeRayTracer rt(cl);
    MPI_Barrier(MPI_COMM_WORLD);
    rt.RenderImage(imagename);
  }
#endif

  GVT_ASSERT(domain_choosen,"The requested domain type is not available, please recompile");

  if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();

  return 0;
}
