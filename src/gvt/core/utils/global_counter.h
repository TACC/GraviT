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
#ifndef GVT_CORE_GLOBAL_COUNTER
#define GVT_CORE_GLOBAL_COUNTER

#include <iostream>
#include <mpi.h>
#include <string>

#if GVT_USE_COUNTER
namespace gvt {
namespace util {
struct global_counter {

  unsigned long local_value = 0;
  std::string text;

  global_counter(std::string t = "Empty") : local_value(0), text(t) {}
  ~global_counter() {}

  void reset() { local_value = 0; }
  void add(long amount) { local_value += amount; }
  void add(int amount) { local_value += amount; }
  void add(std::size_t amount) { local_value += amount; }

  void print() {
    unsigned long global;
    MPI_Reduce(&local_value, &global, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) std::cout << text << "" << global << std::endl;
  }
};
}
}
#else
namespace gvt {
namespace util {
struct global_counter {
  global_counter(std::string t = "Empty") {}
  ~global_counter() {}

  void reset() {}
  void add(long amount) {}
  void add(int amount) {}
  void add(std::size_t amount) {}

  void print() {}
};
}
}

#endif

#endif /* GVT_CORE_GLOBAL_COUNTER */
