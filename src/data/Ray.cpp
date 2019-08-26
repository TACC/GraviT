
#include "Ray.h"
using namespace gvt2;
Ray::Ray(const vec3f& _origin, const vec3f& _direction) {
    origin = vec3f(_origin);
    direction = vec3f(_direction);
    t_min = FLT_MIN;
    t_max = FLT_MAX;
    t = t_min;
    id = 0;
    type = 0;
}
