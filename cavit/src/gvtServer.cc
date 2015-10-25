#include <iostream>
#include <string>
#include <mpi.h>
#include <pthread.h>
#include <vector>
#include <sstream>
#include <stack>

#include <boost/timer/timer.hpp>

#include "gvtState.h"
#include "gvtServer.h"
using namespace std;
using namespace cvt;




void ProcessMessages(StateLocal& state)
{
  //while(1)
  {
    // NetBuffer buffer;
    // buffer._buffer.resize(1024);
   //  MPI_Status status;
   //  MPI_Recv(&buffer._buffer[0], 1024, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, state.intercomm, &status);
   // //printf("server: recieved: %s\n", &buffer._buffer[0]);
   //   if (status.MPI_TAG == stateUniversal.tag)
   //   {
   //  // NetBuffer sendBuffer;
   //  // sendBuffer >> stateUniversal;

   //  NetBuffer sendBuffer;
   //  sendBuffer << stateUniversal;
   //  MPI_Send((void*)&sendBuffer._buffer[0],stateUniversal.GetPackBufferSize(),MPI_BYTE,status.MPI_SOURCE,stateUniversal.tag,stateLocal.intercomm);
    // MPI_Sendv(&stateUniversal.GetPackBuffer()[0],stateUniversal.GetPackBufferSize(),MPI_BYTE,0,TAG_SET_STATE,stateLocal.intercomm, &status);
     // }

  }
  // pthread_exit();
}



        // MPI_Comm globalComm;
//
// server
//
// int main(int argc, char** argv)
void gvtServer::Launch(int argc, char** argv)
{
  std::vector<std::pair<std::string,char*> > ports;
  std::vector<MPI_Comm> portComms;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int commSize;
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  MPI_Comm intercomm;
  MPI_Status status;

// g_granularity = commSize;
  g_granularity = 8;
  LoadBalancer2D<StateTile> baseLoadBalancer(0,0,width,height,g_granularity);

  // MPI_Open_port(MPI::INFO_NULL, port_name);
  // MPI_Publish_name("gvtDisplay", MPI::INFO_NULL, port_name);
  // MPI_Comm_accept(port_name, MPI::INFO_NULL, 0, MPI_COMM_SELF, &stateLocal.intercomm);
  // MPI_Close_port(port_name);
  // printf("display Connected\n");
  //pthread_create(ProcessMessages(state));
  // while(1)

  int numRenderers = commSize-2;  //TODO: DEBUG remore hardcoding!
  #ifdef GVT_USE_PORT_COMM
  ports.push_back(std::pair<std::string, char*>("gvtDisplay",new char[MPI_MAX_PORT_NAME]));
      for (int i =0; i < numRenderers; i++) //TODO: DEBUG BAD DESIGN remove hardcoding
        {cmd_new
         char conName[MPI_MAX_PORT_NAME];
         sprintf(conName, "gvtRenderer%d", i);
         ports.push_back(std::pair<std::string, char*>(std::string(conName),new char[MPI_MAX_PORT_NAME]));
         portComms.push_back(MPI_Comm());
     // MPI_Open_port(MPI::INFO_NULL, port_name);
     // MPI_Publish_name("gvtDisplay", MPI::INFO_NULL, port_name);
     // MPI_Comm_accept(port_name, MPI::INFO_NULL, 0, MPI_COMM_SELF, &stateLocal.intercomm);
     // MPI_Close_port(port_name);
       }

       int commSize;
       {
         for(int i=0;i<ports.size();i++)
         {
          char* port_name = ports[i].second;
          MPI_Open_port(MPI::INFO_NULL, port_name);
          MPI_Publish_name(ports[i].first.c_str(), MPI::INFO_NULL, port_name);
        }
        printf("server waiting for connections\n");
        for (int i =0; i < ports.size(); i++)
        {
          char* port_name = ports[i].second;
          printf("server waiting on %s %s\n", ports[i].first.c_str(), port_name);
          MPI_Comm comm;
          MPI_Comm_accept(port_name, MPI::INFO_NULL, 0, MPI_COMM_SELF, &portComms[i]);
          printf("server accepted conn\n");
          stateLocal.intercomm = portComms[i];
// MPI_Intercomm_merge(comm, 0, &stateLocal.intercomm);
        // MPI_Intercomm_merge(portComms[i], 0, &globalComm);
          MPI_Comm_size(stateLocal.intercomm, &commSize);
          int rsize;
          MPI_Comm_remote_size(stateLocal.intercomm, &rsize);
          printf("server connection established %d remote size:%d\n", commSize-1, rsize);
        }
      }
    #else
      stateLocal.intercomm = MPI_COMM_WORLD;
      #endif  //GVT_USE_PORT_COMM
      {
        MPI_Comm_size(stateLocal.intercomm, &commSize);
        printf("uni size: %d\n", commSize);
        int globalcommSize;
      // // MPI_Comm_size(globalComm, &globalcommSize);
      // printf("global uni size: %d\n", globalcommSize);
        for (int i = 0; i < numRenderers; i++)
        {
          printf("sending client msg\n");
          stateUniversal.Send(i+2, stateLocal.intercomm);
        }
      }


      int size;
      bool done = false;
      StateFrame frame;
      StateMsg msg;
      StateScene scene;
      StateRequest sr;

      LoadBalancer2D<StateTile> loadBalancer(baseLoadBalancer);
      while(!done)
      {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == msg.tag)
        {
          done = true;
        }
        if (status.MPI_TAG == scene.tag)
        {
          scene.Recv(MPI_ANY_SOURCE,stateLocal.intercomm, buffer);
        }
        if (status.MPI_TAG == frame.tag)
        {

          frame.Recv(MPI_ANY_SOURCE, stateLocal.intercomm, buffer);
          if (frame.width != width || frame.height != height)
          {
            width = frame.width;
            height = frame.height;
            baseLoadBalancer = LoadBalancer2D<StateTile>(0,0,width,height,g_granularity);
          }
          loadBalancer = LoadBalancer2D<StateTile>(baseLoadBalancer);
        }
          // bool frameDone = false;
          // int numRenderersToNofity = numRenderers;
     //
     // send tiles
     //
          // while (!frameDone)
        if (status.MPI_TAG == sr.tag)
        {
          sr.Recv(MPI_ANY_SOURCE, stateLocal.intercomm, buffer);
          if (sr.tagr == StateTile::tag || sr.tagr == GVT_WORK_REQUEST)
          {
            StateTile tile = loadBalancer.Next();
        // printf("sending tile %d %d %d %d\n", tile.x, tile.y, tile.width, tile.height);
            tile.Send(sr.status.MPI_SOURCE, stateLocal.intercomm);
          }
        }

          // frame.frame++;
      }
// MPI_Unpublish_name("gvtTest", MPI::INFO_NULL, port_name);
    // MPI_Comm_disconnect(&stateLocal.intercomm);
      MPI_Finalize();
    }


