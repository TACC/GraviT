
#include <gvt/render/data/domain/AbstractDomain.h>

using namespace gvt::render::data::domain;

AbstractDomain::AbstractDomain(gvt::core::math::AffineTransformMatrix<float> m) 
: m(m), domainID(-1), isLoaded(false) 
{
    minv = m.inverse();
    normi = m.upper33().inverse().transpose();
}

AbstractDomain::AbstractDomain(const AbstractDomain &other) 
{
    m = other.m;
    minv = other.minv;
    normi = other.normi;
}

AbstractDomain::~AbstractDomain() 
{}

bool AbstractDomain::intersect(gvt::render::actor::Ray&  r, gvt::render::actor::isecDomList& inter) 
{
    float t;
    if (getWorldBoundingBox().intersectDistance(r, t) && t > gvt::render::actor::Ray::RAY_EPSILON) 
    {
        inter.push_back(gvt::render::actor::isecDom(domainID, t));
        return true;
    }
    return false;
}


// TODO : This code is broken

void AbstractDomain::marchIn(gvt::render::actor::Ray&  ray) 
{
    GVT_ASSERT(false,"TODO: This code is broken");
    gvt::render::data::primitives::Box3D wBox = getWorldBoundingBox();
    float t = FLT_MAX;
    ray.setDirection(-ray.direction);
    while(wBox.inBox(ray.origin)) 
    {
        if(wBox.intersectDistance(ray,t)) ray.origin += ray.direction * t;
        ray.origin += ray.direction * gvt::render::actor::Ray::RAY_EPSILON;
    }
    ray.setDirection(-ray.direction);

}
// TODO : This code is broken
void AbstractDomain::marchOut(gvt::render::actor::Ray&  ray) 
{
    GVT_ASSERT(false,"TODO: This code is broken");
    gvt::render::data::primitives::Box3D wBox = getWorldBoundingBox();
    float t = FLT_MAX;
    //int i =0;
    
    if(wBox.intersectDistance(ray,t)) ray.origin += ray.direction * t;
    while(wBox.intersectDistance(ray,t)) 
    {
        ray.origin += ray.direction * t;
        ray.origin += ray.direction * gvt::render::actor::Ray::RAY_EPSILON;
    }
    ray.origin += ray.direction * gvt::render::actor::Ray::RAY_EPSILON;
}

void AbstractDomain::trace(gvt::render::actor::RayVector& rayList, gvt::render::actor::RayVector& moved_rays) 
{
    GVT_ASSERT(false,"Trace function for this domain was not implemented");
}

bool AbstractDomain::load() 
{
    GVT_ASSERT(false, "Calling domain load generic function\n");
    return false;
}

void AbstractDomain::free() 
{
    GVT_WARNING(false, "Calling domain free generic function\n");
    return;
}

gvt::render::actor::Ray AbstractDomain::toLocal(gvt::render::actor::Ray& r) 
{
    GVT_ASSERT((&r),"NULL POINTER");
    gvt::render::actor::Ray ray(r);
    ray.origin = minv * ray.origin;
    ray.direction = minv * ray.direction;
    return ray;
}

gvt::render::actor::Ray AbstractDomain::toWorld(gvt::render::actor::Ray& r) 
{
    GVT_ASSERT((&r),"NULL POINTER");
    gvt::render::actor::Ray ray(r);
    ray.origin = m * ray.origin;
    ray.direction = m * ray.direction;
    return ray;
}

gvt::core::math::Vector4f AbstractDomain::toLocal(const gvt::core::math::Vector4f& r) 
{
    gvt::core::math::Vector4f ret = minv * r;
    ret.normalize();
    return ret;
}

gvt::core::math::Vector4f AbstractDomain::toWorld(const gvt::core::math::Vector4f& r) 
{
    gvt::core::math::Vector4f ret = m * r;
    ret.normalize();
    return ret;
}

gvt::core::math::Vector4f AbstractDomain::localToWorldNormal(const gvt::core::math::Vector4f &v) 
{
    gvt::core::math::Vector3f ret = normi * (gvt::core::math::Vector3f)v;
    ret.normalize();
    return ret;
}

void AbstractDomain::translate(gvt::core::math::Vector4f t) 
{
    m = m * gvt::core::math::AffineTransformMatrix<float>::createTranslation(t[0],t[1],t[2]);
    
    GVT_DEBUG(DBG_ALWAYS,"M : \n" << m);
    
    minv = m.inverse();
    normi = m.upper33().inverse().transpose();
}


void AbstractDomain::rotate(gvt::core::math::Vector4f t) 
{
    m = m 
            * gvt::core::math::AffineTransformMatrix<float>::createRotation(t[0],1.0,0.0,0.0) 
            * gvt::core::math::AffineTransformMatrix<float>::createRotation(t[1],0.0,1.0,0.0) 
            * gvt::core::math::AffineTransformMatrix<float>::createRotation(t[2],0.0,0.0,1.0);
    minv = m.inverse();
    normi = m.upper33().inverse().transpose();
}

void AbstractDomain::scale(gvt::core::math::Vector4f t) 
{
    m = m * gvt::core::math::AffineTransformMatrix<float>::createScale(t[0],t[1],t[2]);
    minv = m.inverse();
    normi = m.upper33().inverse().transpose();
}


gvt::render::data::primitives::Box3D AbstractDomain::getWorldBoundingBox() 
{
    return getBounds(1);
}

void AbstractDomain::setBoundingBox(gvt::render::data::primitives::Box3D bb) 
{
    boundingBox = bb;
}

gvt::render::data::primitives::Box3D AbstractDomain::getBounds(int type = 0) 
{
    if (type == 0) 
    {
        return boundingBox;
    } 
    else 
    {
        
        gvt::render::data::primitives::Box3D bb; // = boundingBox;
        bb.bounds[0] = m * boundingBox.bounds[0];
        bb.bounds[1] = m * boundingBox.bounds[1];
        return bb;
    }
}

bool AbstractDomain::domainIsLoaded() 
{
    return isLoaded;
}

int AbstractDomain::getDomainID() 
{
    return domainID;
}

void AbstractDomain::setDomainID(int id) 
{
    domainID = id;
}
