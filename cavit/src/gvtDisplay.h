#ifndef GVTDISPLAY_H
#define GVTDISPLAY_H


#include <iostream>
#include <string>
#include <mpi.h>
#include <pthread.h>
#include <vector>
#include <sstream>
#include <stack>

#include <boost/timer/timer.hpp>

#include "gvtState.h"
using namespace std;

namespace cvt
{

class gvtDisplay
{
public:
  gvtDisplay() : width(512), height(512), rank(-1) {}
  void Launch(int argc, char** argv);
  StateLocal stateLocal;
  StateUniversal stateUniversal;
  MPIBuffer buffer;
  int width;
  int height;
  int rank;
  std::vector<StateDomain> domains;
};

}

#endif