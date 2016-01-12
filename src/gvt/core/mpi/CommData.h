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
 * File:   CommData.h
 * Author: jbarbosa
 *
 * Created on January 20, 2014, 5:22 PM
 */

#ifndef GVT_CORE_MPI_COMM_DATA_H
#define GVT_CORE_MPI_COMM_DATA_H

#include <map>
#include <vector>

namespace gvt {
namespace core {
namespace mpi {
/// data struct for MPI-based communication
/** helper struct to organize MPI-based communication.
    \sa
*/
struct comm_data {
  std::vector<int> *to_send;
  std::map<int, int> *data_size;
  std::map<int, int> *local_sorted_doms;
  std::map<int, std::vector<int> > *other_dom;

  MPI_Comm comm;

  unsigned char **in_ray_buf, **local_ray_buf;
  int *in_ray_buf_ptr, *local_offset, *done_flags, *dom_map_buf;

  MPI_Win *win_in_ray_buf, *win_in_ray_buf_ptr, win_done_flags, win_dom_map_buf;

  std::vector<int> to_del;
  int rank, size;
  std::map<int, int> n_index_map;
  int n_size;
  int *n_lut;
  int n_read_rank;
  int *n_write_rank;

  MPI_Comm *comms;
};
}
}
}

#endif /* GVT_CORE_MPI_COMM_DATA_H */
