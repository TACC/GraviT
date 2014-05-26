/* 
 * File:   ray.cpp
 * Author: jbarbosa
 * 
 * Created on March 28, 2014, 1:29 PM
 */

#include "gvt_ray.h"

namespace GVT {
    namespace Data {

        ray::ray(GVT::Math::Point4f origin, GVT::Math::Vector4f direction, float contribution, RayType type, int depth) : 
                type(type), w(contribution), depth(depth) {
            
            this->origin = origin;
            this->direction = (direction).normalize();
            setDirection(direction);
            t =  FLT_MAX;
            tmin = FLT_MAX;
            tmax = -FLT_MAX;
            id = -1;
            tprim = FLT_MAX;
            origin_domain = -1;
        }

        ray::ray(ray &ray, GVT::Math::AffineTransformMatrix<float> &m) {
            origin = m * ray.origin;
            direction = m * (ray.direction).normalize();
            setDirection(direction);
            t = ray.t;
            tmin = ray.tmax;
            tmax = ray.tmax;
            color = ray.color;
            domains = ray.domains;
            id = ray.id;
            r = ray.r;
            b = ray.b;
            type = ray.type;
            w = ray.w;
            depth = ray.depth;
            tprim = ray.tprim;
            origin_domain = ray.origin_domain;
            visited = ray.visited;
        }

        ray::ray(const ray& ray) {
            origin = ray.origin;
            direction = ray.direction;
            inverseDirection = ray.inverseDirection;
            setDirection(direction);
            t = ray.t;
            tmin = ray.tmin;
            tmax = ray.tmax;
            color = ray.color;
            domains = ray.domains;
            id = ray.id;
            r = ray.r;
            b = ray.b;
            w = ray.w;
            type = ray.type;
            depth = ray.depth;
            tprim = ray.tprim;
            origin_domain = ray.origin_domain;
            visited = ray.visited;
        }

        ray::~ray() {
        }

        void ray::setDirection(GVT::Math::Vector4f dir) {
            inverseDirection[3] = 0;
            dir[3] = 0;
            direction = dir.normalize();
            for (int i = 0; i < 3; i++) {
                if (direction[i] != 0) inverseDirection[i] = 1.0 / direction[i];
                else inverseDirection[i] = 0.;
                sign[i] = (inverseDirection[i] < 0);
            }
        }

        void ray::setDirection(double *dir) {
            setDirection(GVT::Math::Vector4f(dir[0], dir[1], dir[2], dir[3]));
        }

        void ray::setDirection(float *dir) {
            setDirection(GVT::Math::Vector4f(dir[0], dir[1], dir[2], dir[3]));
        }

    }
}