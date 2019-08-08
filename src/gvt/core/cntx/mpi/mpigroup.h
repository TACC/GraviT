//
// Created by jbarbosa on 9/6/17.
//

#ifndef CONTEXT_MPIGROUP_H
#define CONTEXT_MPIGROUP_H

#include <cstring>
#include <mpi.h>

namespace cntx {

namespace mpi {
struct MPIGroup {
  MPIGroup(const MPI_Comm c = MPI_COMM_WORLD) : comm(c) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
  }

  MPIGroup duplicate() {
    MPI_Comm a;
    MPI_Comm_dup(comm, &a);
    return MPIGroup(a);
  }

  MPI_Comm comm;
  int rank;
  int size;
};


} // namespace mpi
} // namespace cntx

#endif // CONTEXT_MPIGROUP_H
