

#include <gvt/render/data/Dataset.h>

#include <boost/range/algorithm.hpp>
#include <boost/foreach.hpp>

#ifdef __USE_TAU
#include <TAU.h>
#endif

using namespace gvt::render::data;
using namespace gvt::render::data::domain;
using namespace gvt::render::data::primitives;
using namespace gvt::render::data::scene;
using namespace gvt::render::actor;

bool Dataset::init()
{
    GVT_DEBUG(DBG_ALWAYS, "Dataset::init() abstract load");
    return false;
}

bool Dataset::intersect(Ray&  r, isecDomList& inter)
{
    if (dataSetBB.intersect(r) || dataSetBB.inBox(r))
    {
        r.t = FLT_MAX;
        BOOST_FOREACH(AbstractDomain* d, domainSet) d->intersect(r, inter);
        boost::sort(inter);


//                GVT::Data::isecDomList r; r.assign(inter.rbegin(),inter.rend());
//                inter.clear(); inter.assign(r.begin(),r.end());
        return (!inter.empty());
    }
    return false;
}

AbstractDomain* Dataset::getDomain(int id)
{
#ifdef __USE_TAU
  TAU_START("Dataset::getDomain");
#endif
    GVT_ASSERT_BACKTRACE(id < domainSet.size(),"Getting domain outside bound");
#ifdef __USE_TAU
  TAU_STOP("Dataset::getDomain");
#endif
    return domainSet[id];
}

Light* Dataset::getLight(int id)
{
#ifdef __USE_TAU
  TAU_START("Dataset::getLight");
#endif
    GVT_ASSERT_BACKTRACE(id <lightSet.size(),"Getting light source outside bound");
#ifdef __USE_TAU
  TAU_STOP("Dataset::getLight");
#endif
    return lightSet[id];
}

int Dataset::addDomain(AbstractDomain* dom)
{
#ifdef __USE_TAU
  TAU_START("Dataset::addDomain");
#endif

    dataSetBB.merge(dom->getWorldBoundingBox());
    domainSet.push_back(dom);
    dom->setDomainID(domainSet.size() - 1);

    GVT_DEBUG(DBG_ALWAYS,"Add domain. World BB : " << dataSetBB);

#ifdef __USE_TAU
  TAU_STOP("Dataset::addDomain");
#endif

    return domainSet.size() - 1;
}

int Dataset::addLight(Light* ls)
{
#ifdef __USE_TAU
  TAU_START("Dataset::addLight");
#endif
    dataSetBB.merge(ls->getWorldBoundingBox());
    lightSet.push_back(ls);
#ifdef __USE_TAU
  TAU_STOP("Dataset::addLight");
#endif
    return domainSet.size() - 1;
}

int Dataset::size()
{
    return domainSet.size();
}
