

#include <gvt/render/data/Dataset.h>

#include <boost/range/algorithm.hpp>
#include <boost/foreach.hpp>

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
    GVT_ASSERT_BACKTRACE(id < domainSet.size(),"Getting domain outside bound");
    return domainSet[id];
}

Light* Dataset::getLight(int id) 
{
    GVT_ASSERT_BACKTRACE(id <lightSet.size(),"Getting light source outside bound");
    return lightSet[id];
}

int Dataset::addDomain(AbstractDomain* dom) 
{
    dataSetBB.merge(dom->getWorldBoundingBox());
    domainSet.push_back(dom);
    dom->setDomainID(domainSet.size() - 1);

    GVT_DEBUG(DBG_ALWAYS,"Add domain. World BB : " << dataSetBB);

    return domainSet.size() - 1;
}

int Dataset::addLight(Light* ls) 
{
    dataSetBB.merge(ls->getWorldBoundingBox());
    lightSet.push_back(ls);
    return domainSet.size() - 1;
}

int Dataset::size() 
{
    return domainSet.size();
}
