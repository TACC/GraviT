//
//  GVTTrace.C
//


#include "ConfigFileLoader.h"
//#include "MantaRayTracer.h"
#include "OptixRayTracer.h"

#include <gvt/core/Math.h>
//#include <gvt/render/adapter/manta/Wrapper.h>
//#include <gvt/render/data/Primitives.h>
#include <gvt/render/adapter/optix/Wrapper.h>
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

//    fstream file;
//    file.open(filename.c_str());
//
//    if (!file.good()) {
//        cerr << "ERROR: could not open file '" << filename << "'" << endl;
//        return -1;
//    }
//
//    GVT::Env::RayTracerAttributes& rta = *(GVT::Env::RayTracerAttributes::instance());
//    
//    file >> rta;
//    
//    file.close();
//
//    switch (rta.render_type) {
//        case GVT::Env::RayTracerAttributes::Volume:
//            GVT_DEBUG(DBG_ALWAYS, "Volume dataset");
//            rta.dataset = new GVT::Dataset::Dataset<GVT::Domain::VolumeDomain>(rta.datafile);
//            break;
//        case GVT::Env::RayTracerAttributes::Surface:
//            GVT_DEBUG(DBG_ALWAYS, "Geometry dataset");
//            rta.dataset = new GVT::Dataset::Dataset<GVT::Domain::GeometryDomain>(rta.datafile);
//            break;
//        case GVT::Env::RayTracerAttributes::Manta:
//            rta.dataset = new GVT::Dataset::Dataset<GVT::Domain::MantaDomain>(rta.datafile);
//            break;
//    }
//
//
//    GVT_ASSERT(rta.LoadDataset(), "Unable to load dataset");
//
//    std::cout << rta << std::endl;
//
    //MantaRayTracer rt(&cl.scene);
    OptixRayTracer rt(&cl.scene);
    MPI_Barrier(MPI_COMM_WORLD);
    rt.RenderImage(imagename);

    if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();

    return 0;
}
