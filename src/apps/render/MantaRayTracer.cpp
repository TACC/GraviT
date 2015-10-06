//
//  RayTracer.C
//

#include "MantaRayTracer.h"

#include <gvt/core/mpi/Wrapper.h>
#include <gvt/render/Types.h>
#include <gvt/render/adapter/manta/Wrapper.h>
#include <gvt/render/algorithm/Tracers.h>
#include <gvt/render/data/scene/Camera.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/Schedulers.h>

// Manta includes
#include <Interface/LightSet.h>
#include <Model/Lights/PointLight.h>
#include <Model/Materials/Phong.h>
#include <Model/Readers/PlyReader.h>
// end Manta includes

#include <boost/foreach.hpp>
#include <boost/timer/timer.hpp>

#ifdef PARALLEL
#include <mpi.h>
#endif

using namespace gvtapps::render;
using namespace gvt::core::mpi;
using namespace gvt::render::adapter::manta::data::domain;
using namespace gvt::render::data::domain;
using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;

/// constructor
/**
 * \param cl configuration file loader for ray tracer initalization
 */
MantaRayTracer::MantaRayTracer(ConfigFileLoader& cl) : scene(&cl.scene)
{
    scene->camera.SetCamera(rays,1.0);
    
    //gvt::render::Attributes& rta = *(gvt::render::Attributes::instance());
    gvt::render::RenderContext::CreateContext();
	cntxt = gvt::render::RenderContext::instance();   
	root = cntxt->getRootNode();
	gvt::core::Variant V;
	gvt::core::DBNodeH datanode = cntxt->createNodeFromType("Dataset","somedata",root.UUID());
    //rta.dataset = new gvt::render::data::Dataset();
	gvt::render::data::Dataset* ds = new gvt::render::data::Dataset();
    
    
    BOOST_FOREACH(AbstractDomain* dom, scene->domainSet) 
    {
        GeometryDomain* d = (GeometryDomain*)dom;
        d->setLights(scene->lightSet);
        ds->addDomain(new MantaDomain(d));
        //rta.dataset->addDomain(new MantaDomain(d));
    }

    if (cl.accel_type != ConfigFileLoader::NoAccel)
    {
        std::cout << "creating acceleration structure... ";
        if (cl.accel_type == ConfigFileLoader::BVH)
        {
            //rta.accel_type = gvt::render::Attributes::BVH;
        	//rta.dataset->makeAccel();
        	ds -> makeAccel();
        }
        //rta.dataset->makeAccel(rta);
        std::cout << "...done" << std::endl;
    }
	V = ds;
	datanode["Dataset_Pointer"] = V;
	V = cl.scheduler_type;
	datanode["schedule"] = V;
	V = gvt::render::adapter::Manta;
	datanode["render_type"] = V;
	gvt::core::DBNodeH filmnode = cntxt->createNodeFromType("Film","somefilm",root.UUID());
    V = scene->camera.getFilmSizeWidth();
	filmnode["width"] = V;
	V = scene->camera.getFilmSizeHeight();
	filmnode["height"] = V;
    //rta.view.width = scene->camera.getFilmSizeWidth();
    //rta.view.height = scene->camera.getFilmSizeHeight();
}

/// render the image using the Embree ray tracer
/**
    \param imagename filename for the output image
*/
void MantaRayTracer::RenderImage(std::string imagename = "mpitrace") 
{
    
    boost::timer::auto_cpu_timer t("Total render time: %t\n");
 	int width = gvt::core::variant_toInteger(root["Film"]["width"].value());   
 	int height = gvt::core::variant_toInteger(root["Film"]["height"].value());   
    Image image(width,height, imagename);
    rays = scene->camera.MakeCameraRays();
	int stype = gvt::core::variant_toInteger(root["Dataset"]["schedule"].value());
	if(stype == gvt::render::scheduler::Image) {
    	gvt::render::algorithm::Tracer<ImageScheduler>(rays, image)();  
	} else if (stype == gvt::render::scheduler::Domain) {
    	gvt::render::algorithm::Tracer<DomainScheduler>(rays, image)();  
	}

    gvt::render::algorithm::GVT_COMM mpi;
    if(mpi.root()) image.Write();
    

};

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif




