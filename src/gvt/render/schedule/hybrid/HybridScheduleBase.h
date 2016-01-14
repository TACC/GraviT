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
/*
 * File:   HybridScheduleBase.h
 * Author: jbarbosa
 *
 * Created on January 24, 2014, 2:03 PM
 */

#ifndef GVT_RENDER_SCHEDULE_HYBRID_BYRID_SCHEDULE_BASE_H
#define GVT_RENDER_SCHEDULE_HYBRID_BYRID_SCHEDULE_BASE_H

namespace gvt {
namespace render {
namespace schedule {
namespace hybrid {
/// base for hybrid schedules
/**
\sa AdaptiveSendSchedule, GreedySchedule, LoadAnotherSchedule, LoadAnyOnceSchedule, LoadManySchedule,
LoadOnceSchedule, RayWeightedSpreadSchedule, SpreadSchedule
*/
struct HybridScheduleBase {

  int *newMap;
  int &size;
  int *map_size_buf;
  int **map_recv_bufs;
  int *data_send_buf;

  HybridScheduleBase(int *newMap, int &size, int *map_size_buf, int **map_recv_bufs, int *data_send_buf)
      : newMap(newMap), size(size), map_size_buf(map_size_buf), map_recv_bufs(map_recv_bufs),
        data_send_buf(data_send_buf) {}

  virtual ~HybridScheduleBase() {}

private:
};
}
}
}
}

#endif /* GVT_RENDER_SCHEDULE_HYBRID_BYRID_SCHEDULE_BASE_H */
