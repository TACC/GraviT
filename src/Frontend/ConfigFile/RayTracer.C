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

void RayTracer::RenderImage(GVT::Env::RayTracerAttributes& rta, string imagename = "mpitrace") {
    Image image(rta.view.width, rta.view.height, imagename);
    GVT::Data::RayVector rays;

    GVT::Env::Camera<C_PERSPECTIVE> cam(rays, rta.view, rta.sample_rate);
    cam.MakeCameraRays();


    switch (rta.schedule) {
        case GVT::Env::RayTracerAttributes::Image:
            if (rta.render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, ImageSchedule>(rta, rays, image)();
            else
                GVT::Trace::Tracer<GVT::Domain::VolumeDomain, MPICOMM, ImageSchedule>(rta, rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::Domain:
            if (rta.render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, DomainSchedule>(rta, rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, DomainSchedule>(rta, rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::Greedy:
            if (rta.render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, HybridSchedule<GreedySchedule> >(rta, rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<GreedySchedule> >(rta, rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::Spread:
            if (rta.render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, HybridSchedule<SpreadSchedule> >(rta, rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<SpreadSchedule> >(rta, rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::RayWeightedSpread:
            if (rta.render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, HybridSchedule<RayWeightedSpreadSchedule> >(rta, rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<RayWeightedSpreadSchedule> >(rta, rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::AdaptiveSend:
            if (rta.render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, HybridSchedule<AdaptiveSendSchedule> >(rta, rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<AdaptiveSendSchedule> >(rta, rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::LoadOnce:
            if (rta.render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM,  HybridSchedule<LoadOnceSchedule> >(rta, rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<LoadOnceSchedule> >(rta, rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::LoadAnyOnce:
            if (rta.render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, HybridSchedule<LoadAnyOnceSchedule> >(rta, rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<LoadAnyOnceSchedule> >(rta, rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::LoadAnother:
            if (rta.render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, HybridSchedule<LoadAnotherSchedule> >(rta, rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<LoadAnotherSchedule> >(rta, rays, image)();
            break;
        case GVT::Env::RayTracerAttributes::LoadMany:
            if (rta.render_type == GVT::Env::RayTracerAttributes::Manta)
                GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, HybridSchedule<LoadManySchedule> >(rta, rays, image)();
            else
                GVT::Trace::Tracer< GVT::Domain::VolumeDomain, MPICOMM, HybridSchedule<LoadManySchedule> >(rta, rays, image)();
            break;
        default:
            cerr << "ERROR: unknown schedule '" << rta.schedule << "'" << endl;
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




