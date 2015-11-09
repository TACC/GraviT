/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray
   tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas
   at Austin
   All rights reserved.

   Licensed under the BSD 3-Clause License, (the "License"); you may not use
   this file
   except in compliance with the License.
   A copy of the License is included with this software in the file LICENSE.
   If your copy does not contain the License, you may obtain a copy of the
   License at:

       http://opensource.org/licenses/BSD-3-Clause

   Unless required by applicable law or agreed to in writing, software
   distributed under
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY
   KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under
   limitations under the License.

   GraviT is funded in part by the US National Science Foundation under awards
   ACI-1339863,
   ACI-1339881 and ACI-1339840
   =======================================================================================
   */
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

/// constructor
/**
 * \param cl configuration file loader for ray tracer initalization
 */
EmbreeRayTracer::EmbreeRayTracer(ConfigFileLoader &cl) : scene(&cl.scene) {
  std::cout << "constructing embree ray tracer" << std::endl;
  scene->camera.SetCamera(rays, 1.0);

  gvt::render::Attributes &rta = *(gvt::render::Attributes::instance());

  rta.dataset = new gvt::render::data::Dataset();

  std::cout << "boost foreach creating domains" << std::endl;
  BOOST_FOREACH (AbstractDomain *dom, scene->domainSet) {
    GeometryDomain *d = (GeometryDomain *)dom;
    d->setLights(scene->lightSet);
    rta.dataset->addDomain(new EmbreeDomain(d));
  }

  if (cl.accel_type != ConfigFileLoader::NoAccel) {
    std::cout << "creating acceleration structure... ";
    if (cl.accel_type == ConfigFileLoader::BVH) {
      rta.dataset->makeAccel();
    }
    std::cout << "...done" << std::endl;
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

/// render the image using the Embree ray tracer
/**
    \param imagename filename for the output image
*/
void EmbreeRayTracer::RenderImage(std::string imagename = "mpitrace") {

  std::cout << "rendering image: " << imagename << std::endl;

  boost::timer::auto_cpu_timer t("Total render time: %w\n");

  std::cout << "create image" << std::endl;
  Image image(scene->camera.getFilmSizeWidth(),
              scene->camera.getFilmSizeHeight(), imagename);

  std::cout << "making camera rays" << std::endl;
  rays = scene->camera.MakeCameraRays();
  std::cout << "finished making camera rays" << std::endl;

  std::cout << "calling EmbreeDomain trace/render function" << std::endl;
  gvt::render::algorithm::Tracer<DomainScheduler>(rays, image)();

  gvt::render::algorithm::GVT_COMM mpi;
  if (mpi.root()) {
    std::cout << "writing image to disk" << std::endl;
    image.Write();
  }
};

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif
