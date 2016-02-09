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
//  Attributes.C
//

#include <gvt/render/Attributes.h>

using namespace gvt::render;

#define HACK_TRANSFER_FUNC                                                                                             \
  transfer_func = new unsigned char[4 * 256];                                                                          \
  for (int i = 0; i < 128; ++i) {                                                                                      \
    transfer_func[4 * i + 0] = 0;                                                                                      \
    transfer_func[4 * i + 1] = 2 * i;                                                                                  \
    transfer_func[4 * i + 2] = 255 - 2 * i;                                                                            \
    transfer_func[4 * i + 3] = i;                                                                                      \
  }                                                                                                                    \
  for (int i = 0; i < 128; ++i) {                                                                                      \
    transfer_func[128 + 4 * i + 0] = 2 * i;                                                                            \
    transfer_func[128 + 4 * i + 1] = 255 - 2 * i;                                                                      \
    transfer_func[128 + 4 * i + 2] = 0;                                                                                \
    transfer_func[128 + 4 * i + 3] = 2 * i;                                                                            \
  }

Attributes *Attributes::rta = NULL;

Attributes::Attributes() {
  // topology should be defined in config file
  topology[0] = -1;
  topology[1] = -1;
  topology[2] = -1;

  // XXX TODO: hacked transfer function.  Fix it.
  HACK_TRANSFER_FUNC;

  do_lighting = false;
}

Attributes::Attributes(std::string &datafile_, View &view_, RenderType rt = Volume, ScheduleType st = Image,
                       float rate = 1.f, float ratio = 1.f, float *topo = NULL)
    : view(view_), render_type(rt), schedule(st), sample_rate(rate), sample_ratio(ratio) {
  if (rta != NULL) {
    delete rta;
  }
  rta = this;
  if (topo != NULL) {
    topology[0] = topo[0];
    topology[1] = topo[1];
    topology[2] = topo[2];
  }
  HACK_TRANSFER_FUNC;
  do_lighting = false;
}

Attributes::~Attributes() { delete[] transfer_func; }

namespace gvt {
namespace render {
std::ostream &operator<<(std::ostream &os, Attributes::View const &vi) {
  os << vi.width << " x " << vi.height << ", " << vi.view_angle << " angle\n";
  os << "camera: " << vi.camera[0] << " " << vi.camera[1] << " " << vi.camera[2] << "\n";
  os << " focus: " << vi.focus[0] << " " << vi.focus[1] << " " << vi.focus[2] << "\n";
  os << "    up: " << vi.up[0] << " " << vi.up[1] << " " << vi.up[2] << "\n";

  return os;
}

std::istream &operator>>(std::istream &is, Attributes::View &vi) {
  is >> vi.width >> vi.height;
  is >> vi.view_angle;
  is >> vi.camera[0] >> vi.camera[1] >> vi.camera[2];
  is >> vi.focus[0] >> vi.focus[1] >> vi.focus[2];
  is >> vi.up[0] >> vi.up[1] >> vi.up[2];
  return is;
}

std::ostream &operator<<(std::ostream &os, Attributes const &rta) {
  os << rta.view;
  os << "render type: ";
  switch (rta.render_type) {
  case Attributes::Volume:
    os << "Volume";
    break;
  case Attributes::Surface:
    os << "Surface";
    break;
  case Attributes::Manta:
    os << "Surface";
    break;
  default:
    os << "Unknown (" << rta.render_type << ")";
  }
  os << "\n";

  os << "schedule: ";
  switch (rta.schedule) {
  case Attributes::Image:
    os << "Image";
    break;
  case Attributes::Domain:
    os << "Domain";
    break;
  case Attributes::Greedy:
    os << "Greedy (deprecated)";
    break;
  case Attributes::Spread:
    os << "Spread (deprecated)";
    break;
  case Attributes::RayWeightedSpread:
    os << "RayWeightedSpread (deprecated, use LoadOnce)";
    break;
  case Attributes::AdaptiveSend:
    os << "AdaptiveSend (deprecated)";
    break;
  case Attributes::LoadOnce:
    os << "LoadOnce";
    break;
  case Attributes::LoadAnyOnce:
    os << "LoadAnyOnce";
    break;
  case Attributes::LoadAnother:
    os << "LoadAnother";
    break;
  case Attributes::LoadMany:
    os << "LoadMany (beta)";
    break;
  default:
    os << "Unknown (" << rta.schedule << ")";
  }
  os << "\n";

  os << "sample rate: " << rta.sample_rate;

  os << "dataset: " << rta.dataset << "\n";

  return os;
}

std::istream &operator>>(std::istream &is, Attributes &rta) {
  is >> rta.view;

  std::string rt;
  is >> rt;
  if (rt.find("Volume") != std::string::npos) {
    rta.render_type = Attributes::Volume;
  } else if (rt.find("Surface") != std::string::npos) {
    rta.render_type = Attributes::Surface;
  } else if (rt.find("Manta") != std::string::npos) {
    rta.render_type = Attributes::Manta;
  } else {
    GVT_DEBUG(DBG_ALWAYS, "Unknown render type '" << rt << "', defaulting to Volume");
    rta.render_type = Attributes::Volume;
  }

  std::string sch;
  is >> sch;
  if (sch.find("Image") != std::string::npos)
    rta.schedule = Attributes::Image;
  else if (sch.find("Domain") != std::string::npos)
    rta.schedule = Attributes::Domain;
  else if (sch.find("Greedy") != std::string::npos)
    rta.schedule = Attributes::Greedy;
  else if (sch.find("Spread") != std::string::npos)
    rta.schedule = Attributes::Spread;
  else if (sch.find("RayWeightedSpread") != std::string::npos)
    rta.schedule = Attributes::RayWeightedSpread;
  else if (sch.find("AdaptiveSend") != std::string::npos)
    rta.schedule = Attributes::AdaptiveSend;
  else if (sch.find("LoadOnce") != std::string::npos)
    rta.schedule = Attributes::LoadOnce;
  else if (sch.find("LoadAnyOnce") != std::string::npos)
    rta.schedule = Attributes::LoadAnyOnce;
  else if (sch.find("LoadAnother") != std::string::npos)
    rta.schedule = Attributes::LoadAnother;
  else if (sch.find("LoadMany") != std::string::npos)
    rta.schedule = Attributes::LoadMany;
  else {
    GVT_DEBUG(DBG_ALWAYS, "Unknown schedule '" << sch << "', defaulting to Image");
    rta.schedule = Attributes::Image;
  }

  is >> rta.sample_rate >> rta.sample_ratio;

  is >> rta.topology[0] >> rta.topology[1] >> rta.topology[2];

  is >> rta.datafile;

  return is;
}
}
} // namespace render } namespace gvt }