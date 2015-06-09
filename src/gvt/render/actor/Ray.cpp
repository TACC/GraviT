/*
 * File:   Ray.cpp
 * Author: jbarbosa
 *
 * Created on March 28, 2014, 1:29 PM
 */

#include <gvt/render/actor/Ray.h>

#include <boost/foreach.hpp>
#include <boost/pool/singleton_pool.hpp>
#include <boost/pool/pool_alloc.hpp>

using namespace gvt::core::math;
using namespace gvt::render::actor;

const float Ray::RAY_EPSILON = 1.e-6;

Ray::Ray(Point4f origin, Vector4f direction, float contribution, RayType type, int depth)
: type(type), w(contribution), depth(depth)
{

    this->origin = origin;
    this->direction = (direction).normalize();
    setDirection(direction);
    t = FLT_MAX;
            //            tmin = FLT_MAX;
            //            tmax = -FLT_MAX;
    id = -1;
            //            tprim = FLT_MAX;
            //            origin_domain = -1;
}

Ray::Ray(Ray &ray, AffineTransformMatrix<float> &m)
{
    origin = m * ray.origin;
    direction = m * (ray.direction).normalize();
    setDirection(direction);
    t = ray.t;
            //            tmin = ray.tmax;
            //            tmax = ray.tmax;
    color = ray.color;
    domains = ray.domains;
    id = ray.id;
            //            r = ray.r;
            //            b = ray.b;
    type = ray.type;
    w = ray.w;
    depth = ray.depth;
            //            tprim = ray.tprim;
            //            origin_domain = ray.origin_domain;
            //            visited = ray.visited;
}

Ray::Ray(const Ray& ray)
{
    origin = ray.origin;
    direction = ray.direction;
    inverseDirection = ray.inverseDirection;
    setDirection(direction);
    t = ray.t;
            //            tmin = ray.tmin;
            //            tmax = ray.tmax;
    color = ray.color;
    domains = ray.domains;
    id = ray.id;
            //            r = ray.r;
            //            b = ray.b;
    w = ray.w;
    type = ray.type;
    depth = ray.depth;
            //            tprim = ray.tprim;
            //            origin_domain = ray.origin_domain;
            //            visited = ray.visited;
}

Ray::~Ray()
{
}

Ray::Ray(const unsigned char* buf)
{
    GVT_DEBUG(DBG_ALWAYS, "in Ray::Ray(const unsigned char* buf)");
    origin = Vector4f((float*) buf);
    buf += origin.packedSize();
    direction = Vector4f((float*) buf);
    buf += direction.packedSize();
    id = *((int*) buf);
    buf += sizeof (int);
    depth = *((int*) buf);
    buf += sizeof (int);
    type = *((int*) buf);
    buf += sizeof (int);
            //            r = *((double*) buf);
            //            buf += sizeof (double);
    w = *((double*) buf);
    buf += sizeof (double);
    t = *((double*) buf);
            //            buf += sizeof (double);
            //            tmax = *((double*) buf);
    buf += sizeof (double);
    color = GVT_COLOR_ACCUM(buf);
    buf += color.packedSize();
    int domain_size = *((int*) buf);
    buf += sizeof (int);
    for (int i = 0; i < domain_size; ++i, buf += sizeof (isecDom))
    {
                //domains.insert(isecDom(*(float*) buf,*(float*) (buf + sizeof (float))));
        domains.push_back(isecDom(*(float*) buf));
    }

}

int Ray::packedSize()
{
    int total_size = origin.packedSize() + direction.packedSize() + color.packedSize();
    total_size = 4 * sizeof (int) + 4 * sizeof (double);
    total_size = sizeof (isecDom) * domains.size();
    return total_size;
}

int Ray::pack(unsigned char* buffer)
{

    unsigned char* buf = buffer;

    buffer += origin.pack(buf);
    buffer += direction.pack(buf);
    *((int*) buf) = id;
    buf += sizeof (int);
    *((int*) buf) = depth;
    buf += sizeof (int);
    *((int*) buf) = type;
    buf += sizeof (int);
            //            *((double*) buf) = r;
            //            buf += sizeof (double);
    *((double*) buf) = w;
    buf += sizeof (double);
    *((double*) buf) = t;
    buf += sizeof (double);
            //            *((double*) buf) = tmax;
            //            buf += sizeof (double);
    buf += color.pack(buf);
    *((int*) buf) = domains.size();
    buf += sizeof (int);

    BOOST_FOREACH(isecDom& d, domains)
    {
                //
                //                *(float*) buf = boost::get<0>(d);
                //                buf += sizeof (float);
                //                *(int*) buf = boost::get<1>(d);
                //                buf += sizeof (int);

        *(isecDom*) buf = d;
        buf += sizeof (int);
    }

            //                for (int i = 0; i < domains.size(); ++i, buf += (sizeof (int) + sizeof(float))))
            //                    *((int*) buf) = domains;

    return packedSize();
}

void Ray::setDirection(Vector4f dir)
{
    inverseDirection[3] = 0;
    dir[3] = 0;
    direction = dir.normalize();
    for (int i = 0; i < 3; i++)
    {
        if (direction[i] != 0) inverseDirection[i] = 1.0 / direction[i];
        else inverseDirection[i] = 0.;
    }
}

void Ray::setDirection(double *dir)
{
    setDirection(Vector4f(dir[0], dir[1], dir[2], dir[3]));
}

void Ray::setDirection(float *dir)
{
    setDirection(Vector4f(dir[0], dir[1], dir[2], dir[3]));
}
