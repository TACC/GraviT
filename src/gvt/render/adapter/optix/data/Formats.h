/*
 * File:   optixdata.h
 * Author: jbarbosa
 *
 * Created on January 11, 2015, 10:24 PM
 */

#ifndef GVT_RENDER_ADAPTER_OPTIX_DATA_FORMATS_H
#define GVT_RENDER_ADAPTER_OPTIX_DATA_FORMATS_H

namespace gvt {
namespace render {
namespace adapter {
namespace optix {
namespace data {

/// OptiX ray format
struct OptixRay {
  float origin[3];
  float t_min;
  float direction[3];
  float t_max;
  friend std::ostream &operator<<(std::ostream &os, const OptixRay &r) {
    return (os << "ray  o: " << r.origin[0] << ", " << r.origin[1] << ", "
               << r.origin[2] << " d: " << r.direction[0] << ", "
               << r.direction[1] << ", " << r.direction[2]);
  }
};

/// OptiX hit format
struct OptixHit {
  float t;
  int triangle_id;
  float u;
  float v;
  friend std::ostream &operator<<(std::ostream &os, const OptixHit &oh) {
    return (os << "hit  t: " << oh.t << " triID: " << oh.triangle_id);
  }
};
}
}
}
}
}

#endif /* GVT_RENDER_ADAPTER_OPTIX_DATA_FORMATS_H */
