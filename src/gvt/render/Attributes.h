/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas at Austin
   All rights reserved.

   Licensed under the BSD 3-Clause License, (the "License"); you may not use this file
   except in compliance with the License.
   A copy of the License is included with this software in the file LICENSE.
   If your copy does not contain the License, you may obtain a copy of the License at:

       http://opensource.org/licenses/BSD-3-Clause

   Unless required by applicable law or agreed to in writing, software distributed under
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.
   See the License for the specific language governing permissions and limitations under
   limitations under the License.

   GraviT is funded in part by the US National Science Foundation under awards ACI-1339863,
   ACI-1339881 and ACI-1339840
   ======================================================================================= */
//
//  Attributes.h
//

#ifndef GVT_RENDER_ATTRIBUTES_H
#define GVT_RENDER_ATTRIBUTES_H

#include <gvt/core/Math.h>
#include <gvt/core/Types.h>
#include <gvt/render/data/Dataset.h>
#include <gvt/render/data/Primitives.h>

#include <iostream>

namespace gvt {
namespace render {
/// GraviT internal state for ray tracer. Deprecated for RenderContext?
class Attributes {
public:
  static Attributes *rta;

  class View {
  public:
    View() {}

    View(const View &v) {
      width = v.width;
      height = v.height;
      view_angle = v.view_angle;
      camera[0] = v.camera[0];
      camera[1] = v.camera[1];
      camera[2] = v.camera[2];
      focus[0] = v.focus[0];
      focus[1] = v.focus[1];
      focus[2] = v.focus[2];
      up[0] = v.up[0];
      up[1] = v.up[1];
      up[2] = v.up[2];
    }

    friend std::ostream &operator<<(std::ostream &os, View const &vi);
    friend std::istream &operator>>(std::istream &, View &);

    int width, height;
    int view_angle;
    gvt::core::math::Point4f camera;
    gvt::core::math::Point4f focus;
    gvt::core::math::Vector4f up;

    float parallelScale;
    float nearPlane;
    float farPlane;
  };

  // clang-format off
  enum RenderType {
    Volume,
    Surface,
    Manta,
    Optix
  };
  // clang-format on

  enum ScheduleType {
    Image,
    Domain,
    Greedy,            // PAN: deprecated
    Spread,            // PAN: deprecated
    RayWeightedSpread, // PAN: from EGPGV 2012 paper, deprecated, now called LoadOnce
    AdaptiveSend,      // PAN: deprecated
    LoadOnce,          // PAN: from TVCG 2013 paper
    LoadAnyOnce,       // PAN: from TVCG 2013 paper
    LoadAnother,       // PAN: from TVCG 2013 paper
    LoadMany
  };

  // clang-format off
  enum AccelType {
    NoAccel,
    BVH
  };
  // clang-format on

  Attributes();
  Attributes(std::string &, View &, RenderType, ScheduleType, float, float, float *);
  ~Attributes();

  static Attributes *instance() {
    if (!rta) rta = new Attributes();
    return rta;
  }

  bool LoadDataset() {
    GVT_DEBUG(DBG_ALWAYS, "Sent load");
    return dataset->init();
  }

  ScheduleType GetSchedule() { return schedule; }

  AccelType GetAccelType() { return accel_type; }

  unsigned char *GetTransferFunction() { return transfer_func; }

  int *GetTopology() { return topology; }

  void GetVarExtents(float &vmin, float &vmax) {
    vmin = 1.09554;
    vmax = 5.88965;
  } // XXX TODO: hacked for noise.conf

  void GetOpacityExtents(float &omin, float &omax) {
    omin = 1.09554;
    omax = 5.88965;
  } // XXX TODO: hacked for noise.conf

  void SetLightingFlag(bool doit) { do_lighting = doit; }

  bool GetLightingFlag() { return do_lighting; }

  float GetReflectivityThreshold() { return 1.; }

  friend std::ostream &operator<<(std::ostream &, Attributes const &);
  friend std::istream &operator>>(std::istream &, Attributes &);
  friend class RayTracer;

public:
  View view;
  RenderType render_type;
  ScheduleType schedule;
  AccelType accel_type;
  float sample_rate;
  float sample_ratio;
  gvt::render::data::Dataset *dataset;
  std::string datafile;
  int topology[3];
  unsigned char *transfer_func;

  bool do_lighting;
};

typedef Attributes RTA;
}
}
#endif // GVT_RENDER_ATTRIBUTES_H
