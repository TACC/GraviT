#include <iostream>
#include <string>
#include <mpi.h>
using namespace std;

//
// client
//
// int main(int argc, char** argv)
void runClient()
{
  char port_name[MPI_MAX_PORT_NAME];
  // char buf[256];
  string msg("hello");
  // if (argc > 1)
    // msg = string(argv[1]);
  MPI_Comm intercomm, parentcomm;
  MPI_Status status;
  int errcodes[1];
  MPI_Init(0, 0);
  MPI_Comm_get_parent(&parentcomm);
  if (parentcomm == MPI_COMM_NULL)
  {
    MPI_Comm_spawn("gvtClient", MPI_ARGV_NULL, 1, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &intercomm, errcodes);
    printf("parent: spawned children\n");
  } else
  {
    printf("child: I was spawned\n");
    MPI_Lookup_name("gvtTest", MPI_INFO_NULL, port_name);
    MPI_Comm_connect(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF, &intercomm);
    printf("client sending msg\n");
    MPI_Send((void*)msg.c_str(), msg.length(), MPI_CHAR, 0, 0, intercomm);
    printf("client msg sent\n");
    MPI_Comm_disconnect(&intercomm);
  }
  MPI_Finalize();
  // return 0;
}

//
// runDisplay
//
// int main(int argc, char** argv)
void runDisplay()
{
  char port_name[MPI_MAX_PORT_NAME];
  // char buf[256];
  string msg("hello");
  // if (argc > 1)
    // msg = string(argv[1]);
  MPI_Comm intercomm, parentcomm;
  MPI_Status status;
  int errcodes[1];
  //MPI_Init(0, 0);
  MPI_Comm_get_parent(&parentcomm);
  if (parentcomm == MPI_COMM_NULL)
  {
    MPI_Comm_spawn("gvtDisplay", MPI_ARGV_NULL, 1, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &intercomm, errcodes);
    printf("parent: spawned display\n");
  } else
  {
    printf("child: I was spawned\n");
    MPI_Lookup_name("gvtTest", MPI_INFO_NULL, port_name);
    MPI_Comm_connect(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF, &intercomm);
    printf("client sending msg\n");
    MPI_Send((void*)msg.c_str(), msg.length(), MPI_CHAR, 0, 0, intercomm);
    printf("client msg sent\n");
    MPI_Comm_disconnect(&intercomm);
  }
  MPI_Finalize();
  // return 0;
}

void gvtInit()
{
   runClient();
  // runDisplay();
}

