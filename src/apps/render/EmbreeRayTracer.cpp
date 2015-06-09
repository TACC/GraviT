//
//  RayTracer.C
//

#include "EmbreeRayTracer.h"

#include <gvt/core/mpi/Wrapper.h>
#include <gvt/render/adapter/embree/Wrapper.h>
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
using namespace gvt::render::adapter::embree::data::domain;
using namespace gvt::render::data::domain;
using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;

EmbreeRayTracer::EmbreeRayTracer(gvt::render::data::Dataset* scene) : scene(scene)
{
    std::cout << "constructing embree ray tracer" << std::endl;
    scene->camera.SetCamera(rays,1.0);

    gvt::render::Attributes& rta = *(gvt::render::Attributes::instance());

    rta.dataset = new gvt::render::data::Dataset();


    std::cout << "boost foreach creating domains" << std::endl;
    BOOST_FOREACH(AbstractDomain* dom, scene->domainSet)
    {
        GeometryDomain* d = (GeometryDomain*)dom;
        d->setLights(scene->lightSet);
        rta.dataset->addDomain(new EmbreeDomain(d));
    }


    std::cout << "setting ray attributes" << std::endl;
    rta.view.width = scene->camera.getFilmSizeWidth();
    rta.view.height = scene->camera.getFilmSizeHeight();
    rta.view.camera = scene->camera.getEye();
    rta.view.focus = scene->camera.getLook();
    rta.view.up = scene->camera.up;

    rta.sample_rate = 1.0f;
    rta.sample_ratio = 1.0f;

    rta.do_lighting = true;
    rta.schedule = gvt::render::Attributes::Image;
    rta.render_type = gvt::render::Attributes::Manta;

    rta.datafile = "";
    
    std::cout << "finished constructing EmbreeRayTracer" << std::endl;
}

void EmbreeRayTracer::RenderImage(std::string imagename = "mpitrace")
{

   
    std::cout << "rendering image: " << imagename << std::endl;

    boost::timer::auto_cpu_timer t("Total render time: %w\n");

    std::cout << "create image" << std::endl;
    Image image(scene->camera.getFilmSizeWidth(),scene->camera.getFilmSizeHeight(), imagename);

    std::cout << "making camera rays" << std::endl;
    rays = scene->camera.MakeCameraRays();
    std::cout << "finished making camera rays" << std::endl;

    std::cout << "calling EmbreeDomain trace/render function" << std::endl;
    gvt::render::algorithm::Tracer<DomainScheduler>(rays, image)();
    
    gvt::render::algorithm::GVT_COMM mpi;
    if(mpi.root()) {
        std::cout << "writing image to disk" << std::endl;
        image.Write();
    }

};

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif

