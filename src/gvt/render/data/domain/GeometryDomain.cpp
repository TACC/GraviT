//
// GeometryDomain.C
//


#include <gvt/render/data/domain/GeometryDomain.h>

#include <gvt/render/data/domain/reader/PlyReader.h>

#include <boost/timer/timer.hpp>

#ifdef __USE_TAU
#include <TAU.h>
#endif

using namespace gvt::render::data::domain;
using namespace gvt::render::data::primitives;
using namespace gvt::render::data::scene;
using namespace gvt::core::math;

void GeometryDomain::free()
{
#ifdef __USE_TAU
  TAU_START("GeometryDomain::free");
#endif
    if (!isLoaded) return;
    for (int i = lights.size() - 1; i >= 0; i--)
    {
        delete lights[i];
        lights.pop_back();
    }
    if (mesh)
    {
        delete mesh;
        mesh = NULL;
    }
    isLoaded = false;
#ifdef __USE_TAU
  TAU_STOP("GeometryDomain::free");
#endif

}

bool GeometryDomain::load()
{
#ifdef __USE_TAU
  TAU_START("GeometryDomain::load");
#endif

    if (isLoaded) return true;
    GVT_ASSERT(filename == "", "No filename");
    {
      GVT_DEBUG(DBG_LOW,"GeometryDomain::load() loading ply file");
      boost::timer::auto_cpu_timer t;
      gvt::render::data::domain::reader::PlyReader reader(filename);
      mesh = reader.getMesh();
    }

    lights.push_back(new PointLight(Point4f(5.0, 5.0, 5.0, 1.f), Color(1.f, 1.f, 1.f, 1.f)));
    mesh->setMaterial(new Lambert(Color(1.f, .0f, .0f, 1.f)));
    boundingBox = *(mesh->getBoundingBox());
    isLoaded = true;
#ifdef __USE_TAU
  TAU_STOP("GeometryDomain::load");
#endif

    return isLoaded;
}

namespace gvt{ namespace render{ namespace data{ namespace domain{
std::ostream&
operator<<(std::ostream& os, GeometryDomain const& d)
{
    os << "geometry domain @ addr " << (void*) &d << std::endl;
    os << "    XXX not yet implemented XXX" << std::endl;
    os << std::flush;

    return os;
}
}}}} // namepsace domain} namespace data} namespace render} namespace gvt}
