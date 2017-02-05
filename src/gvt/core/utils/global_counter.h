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
