//
//  RayTracer.C
//
#include "EmbreeRayTracer.h"
#include "OptixRayTracer.h"

#include <gvt/core/mpi/Wrapper.h>
#include <gvt/render/adapter/optix/Wrapper.h>
#include <gvt/render/algorithm/Tracers.h>
#include <gvt/render/data/scene/Camera.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/Schedulers.h>

#include <boost/foreach.hpp>

#ifdef PARALLEL
#include <mpi.h>
#endif

using namespace gvtapps::render;
using namespace gvt::core::mpi;
using namespace gvt::render::adapter::optix::data::domain;
using namespace gvt::render::data::domain;
using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;

OptixRayTracer::OptixRayTracer(ConfigFileLoader& cl) : scene(&cl.scene)
{
    scene->camera.SetCamera(rays,1.0);
    
    gvt::render::Attributes& rta = *(gvt::render::Attributes::instance());
    
    rta.dataset = new gvt::render::data::Dataset();
    
    
    BOOST_FOREACH(AbstractDomain* dom, scene->domainSet) 
    {
        GeometryDomain* d = (GeometryDomain*)dom;
        d->setLights(scene->lightSet);
        rta.dataset->addDomain(new OptixDomain(d));
    }

    if (cl.accel_type != ConfigFileLoader::NoAccel)
    {
        std::cout << "creating acceleration structure... ";
        if (cl.accel_type == ConfigFileLoader::BVH)
        {
        	rta.dataset->makeAccel();
        }
        std::cout << "...done" << std::endl;
    }

    rta.view.width = scene->camera.getFilmSizeWidth();
    rta.view.height = scene->camera.getFilmSizeHeight();
    rta.view.camera = scene->camera.getEye();
    rta.view.focus = scene->camera.getLook();
    rta.view.up = scene->camera.up;
    
    rta.sample_rate = 1.0f;
    rta.sample_ratio = 1.0f;
    
    rta.do_lighting = true;
    rta.schedule = gvt::render::Attributes::Image;
    rta.render_type = gvt::render::Attributes::Optix;
    
    rta.datafile = "";
}

void OptixRayTracer::RenderImage(std::string imagename = "mpitrace") 
{
    
    boost::timer::auto_cpu_timer t("Total render time: %t\n");
    
    Image image(scene->camera.getFilmSizeWidth(),scene->camera.getFilmSizeHeight(), imagename);
    rays = scene->camera.MakeCameraRays();
    gvt::render::algorithm::Tracer<DomainScheduler>(rays, image)();  
    gvt::render::algorithm::GVT_COMM mpi;
    if(mpi.root()) image.Write();

};

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif




