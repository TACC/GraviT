//
//  RayTracer.C
//

#include "RayTracer.h"
#include <Model/Materials/Phong.h>
#include <Model/Readers/PlyReader.h>
#include <Interface/LightSet.h>
#include <Model/Lights/PointLight.h>

#include <GVT/Environment/Camera.h>
#include <GVT/Tracer/tracers.h>
#include <GVT/MPI/mpi_wrappers.h>
#include <Backend/Manta/gvtmanta.h>
//#include <assert.h>

#ifdef PARALLEL
#include <mpi.h>
#endif

#include <GVT/Scheduler/schedulers.h>

void RayTracer::RenderImage(string imagename = "mpitrace") {
    
    boost::timer::auto_cpu_timer t("Total render time: %t\n");
    
    Image image(GVT::Env::RayTracerAttributes::rta->view.width, GVT::Env::RayTracerAttributes::rta->view.height, imagename);
   
    GVT::Env::Camera<C_PERSPECTIVE> cam(rays, GVT::Env::RayTracerAttributes::rta->view, GVT::Env::RayTracerAttributes::rta->sample_rate);
    cam.MakeCameraRays();

    int render_type = GVT::Env::RayTracerAttributes::rta->render_type;
    

    switch (GVT::Env::RayTracerAttributes::rta->schedule) {
        case GVT::Env::RayTracerAttributes::Image:
            if (render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, ImageSchedule>(rays, image)();
            else
                GVT::Trace::Tracer<GVT::Domain::VolumeDomain, MPICOMM, ImageSchedule>(rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::Domain:
            if (render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, DomainSchedule>(rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, DomainSchedule>(rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::Greedy:
            if (render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, HybridSchedule<GreedySchedule> >(rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<GreedySchedule> >(rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::Spread:
            if (render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, HybridSchedule<SpreadSchedule> >(rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<SpreadSchedule> >(rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::RayWeightedSpread:
            if (render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, HybridSchedule<RayWeightedSpreadSchedule> >(rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<RayWeightedSpreadSchedule> >(rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::AdaptiveSend:
            if (render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, HybridSchedule<AdaptiveSendSchedule> >(rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<AdaptiveSendSchedule> >(rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::LoadOnce:
            if (render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM,  HybridSchedule<LoadOnceSchedule> >(rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<LoadOnceSchedule> >(rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::LoadAnyOnce:
            if (render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, HybridSchedule<LoadAnyOnceSchedule> >(rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<LoadAnyOnceSchedule> >(rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::LoadAnother:
            if (render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, HybridSchedule<LoadAnotherSchedule> >(rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<LoadAnotherSchedule> >(rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::LoadMany:
            if (render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, HybridSchedule<LoadManySchedule> >(rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<LoadManySchedule> >(rays, image)();
            break;
        default:
            cerr << "ERROR: unknown schedule '" << GVT::Env::RayTracerAttributes::rta->schedule << "'" << endl;
            return;
    }


    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        image.Write();
    }

};

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif




