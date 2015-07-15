#include <iostream>
#include <string>
#include <mpi.h>
#include <pthread.h>
#include <vector>
#include <sstream>
#include <stack>

#include "gvtState.h"
#include "gvtServer.h"
#include "gvtDisplay.h"
#include "gvtWorker.h"

using namespace cvt;

int g_width =512, g_height=512;

int main(int argc, char** argv)
{
// char buf[256];
    string msg("hello");
  // if (argc > 1)
  //   msg = string(argv[1]);
    printf("client spawned\n");
    MPI_Comm parentcomm;
    int errcodes[1];
    MPI_Init(&argc, &argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD  , &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  // boost::mpi::environment env;
  // boost::mpi::communicator world;
    MPI_Comm_get_parent(&parentcomm);
      // if (parentcomm == MPI_COMM_NULL)

    std::vector<StateDomain> doms;
  if(rank == 0) //gvtDisplay for now
  {

    glm::vec3 min(-50,-50,0);
      for(int i=0; i < 10;i++)
  {
    for (int j=0;j < 10;j++)
    {
      for(int k=0;k<1;k++)
      {
        StateDomain dom;
        dom.bound_min = glm::vec3(i*10+min[0],j*10+min[1],k*10+min[2]);
        dom.bound_max = glm::vec3(i*10+min[0]+9,j*10+min[1]+9,k*10+min[2]+9);
        dom.id = k*100+j*10+i;
        doms.push_back(dom);
      }
    }
  }
  // StateDomain dom;
  //   dom.bound_min = glm::vec3(0,0,0);
  //   dom.bound_max = glm::vec3(20,10,10);
  //   dom.id = 10;
  //   doms.push_back(dom);

    gvtDisplay display;
    display.domains = doms;
    display.width = g_width;
    display.height = g_height;
    display.Launch(argc, argv);
  }
  else if (rank == 1)
  {
    gvtServer server;
    server.width = g_width;
    server.height = g_height;
    server.Launch(argc, argv);
  } else //renderer
  {
    Worker worker;
    worker.Launch(argc, argv);
  }

MPI_Finalize();
return 0;
}
