//
// Utils.C
//

#include "Utils.h"

#include <cfloat>

bool
X_Box( const GVT::Data::ray& r, const float* min, const float* max, 
       float& t_near, float& t_far )
{
    t_near = -FLT_MAX;
    t_far = FLT_MAX;
    
    for (int i=0; i < 3; ++i)
    {
        if ( r.direction[i] == 0)
        {
            if ((r.origin[i] < min[i]) | (r.origin[i] > max[i]))
                return false;
        }
        else
        {
            float inv_d = 1.f / r.direction[i];
            float t1, t2;
            t1 = (min[i] - r.origin[i]) * inv_d;
            t2 = (max[i] - r.origin[i]) * inv_d;
            if (t1 > t2)
            {
                float temp = t1;
                t1 = t2;
                t2 = temp;
            }

            if (t1 > t_near) t_near = t1;
            if (t2 < t_far)  t_far = t2;
            if (t_near > t_far) return false;
            if (t_far < 0) return false;
        }       
    }

    return true;
}
