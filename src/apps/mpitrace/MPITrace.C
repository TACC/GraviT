//
//  MPITrace.C
//

#include <mpi.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

//#include <GVT/MPI/mpi_wrappers.h>
//#include <Frontend/cmd/RayTracer.h>

#include <GVT/Data/primitives.h>
#include <GVT/Math/GVTMath.h>
#include <Frontend/ConfigFile/RayTracer.h>
#include <GVT/Environment/RayTracerAttributes.h>
#include <Backend/Manta/gvtmanta.h>
#include <Frontend/ConfigFile/Dataset/Dataset.h>
using namespace std;



int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);

    string filename, imagename;

    if (argc > 1)
        filename = argv[1];
    else
        filename = "./mpitrace.conf";

    if (argc > 2)
        imagename = argv[2];
    else
        imagename = "MPITrace";

    fstream file;
    file.open(filename.c_str());

    if (!file.good()) {
        cerr << "ERROR: could not open file '" << filename << "'" << endl;
        return -1;
    }

    GVT::Env::RayTracerAttributes& rta = *(GVT::Env::RayTracerAttributes::instance());
    
    file >> rta;
    
    file.close();

    switch (rta.render_type) {
        case GVT::Env::RayTracerAttributes::Volume:
            GVT_DEBUG(DBG_ALWAYS, "Volume dataset");
            rta.dataset = new GVT::Dataset::Dataset<GVT::Domain::VolumeDomain>(rta.datafile);
            break;
        case GVT::Env::RayTracerAttributes::Surface:
            GVT_DEBUG(DBG_ALWAYS, "Geometry dataset");
            rta.dataset = new GVT::Dataset::Dataset<GVT::Domain::GeometryDomain>(rta.datafile);
            break;
        case GVT::Env::RayTracerAttributes::Manta:
            rta.dataset = new GVT::Dataset::Dataset<GVT::Domain::MantaDomain>(rta.datafile);
            break;
    }


    GVT_ASSERT(rta.LoadDataset(), "Unable to load dataset");

    std::cout << rta << std::endl;

    RayTracer rt;
    MPI_Barrier(MPI_COMM_WORLD);
    rt.RenderImage(imagename);

    if (MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();

    return 0;
}
