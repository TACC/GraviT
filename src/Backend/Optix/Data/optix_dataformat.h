/* 
 * File:   optixdata.h
 * Author: jbarbosa
 *
 * Created on January 11, 2015, 10:24 PM
 */

#ifndef OPTIXDATA_H
#define	OPTIXDATA_H


//#include <GVT/Data/primitives/gvt_ray.h>

namespace GVT {

namespace Domain {

struct OptixRayFormat {
  float origin[3];
  float t_min;
  float direction[3];
  float t_max;
};

struct OptixHitFormat {
  float t;
  int triangle_id;
  float u;
  float v;
};

};

};



#endif	/* OPTIXDATA_H */

