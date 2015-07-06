#ifndef GVTSERVER_H
#define GVTSERVER_H

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

template<class T>
struct LoadBalancer
{
  LoadBalancer(size_t begin, size_t end, int granularity_)
  : granularity(granularity_)
  {
    if (granularity < 1)
      granularity = 1;
    Reset();
  }
  void Reset()
  {
    size_t step = max(size_t(1),(end-begin-1)/granularity);
    // size_t stepw = steph = sqrt(step);
    for (size_t b = begin; b < end; b+=step )
    {
      T tile(b, b+step-1);
      tiles.push(tile);
      // printf("load balancer adding tile %d %d\n", tile.begin, tile.end);
    }
  }

  T Next()
  {
    T tile;
    if (!tiles.empty())
    {
      tile = tiles.top();
      tiles.pop();
    }
    return tile;
  }
  int granularity;
  size_t begin, end;
  stack<T> tiles;
};

template<class T>
struct LoadBalancer2D
{
  LoadBalancer2D(int x_, int y_, int width_, int height_, int granularity_)
  : x(x_),y(y_),width(width_),height(height_),granularity(granularity_)
  {
    if (granularity < 1)
      granularity = 1;
    size_t step = max(1,width*height);
    int stepw,steph;
    // stepw = width/sqrt(granularity);
    // steph = height*float(height)/float(width)/sqrt(granularity);
    // stepw = steph = width*height/granularity;
    stepw = width/granularity;
    steph = height/granularity;
    // for (size_t b = begin; b < end; b+=step )
    int ty = y;
    for(;ty<y+height;ty+=steph)
    {
      int tx = x;
      for(;tx < x+width; tx += stepw)
      {
        int twidth = min(stepw,(x+width)-tx);
        int theight = min(steph,(y+height)-ty);
        StateTile tile(tx,ty,twidth,theight);
        tiles.push(tile);
        // printf("load balancer adding tile %d %d\n", tile.x, tile.y);
      }
    }
    // printf("loadBalancer2D : stepwh: %d %d tiles: %d\n", stepw, steph, tiles.size());
  }
  T Next()
  {
    T tile;
    if (!tiles.empty())
    {
      tile = tiles.top();
      tiles.pop();
    }
    return tile;
  }
  int granularity;
  int x,y,width, height;
  stack<T> tiles;
};

class gvtServer
{
public:
  gvtServer() : g_granularity(1) {}
void Launch(int argc, char** argv);
  StateLocal stateLocal;
StateUniversal stateUniversal;
MPIBuffer buffer;
int width;
int height;
// int width = 512;
// int height = 512;
int g_granularity;
};
        // MPI_Comm globalComm;
//
// server
//
// int main(int argc, char** argv)
// {
// }

}

#endif
