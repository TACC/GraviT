//
//  RayTracer.h
//


#ifndef GVT_RAY_TRACER_H
#define GVT_RAY_TRACER_H

#include <GVT/Domain/domains.h>
#include <GVT/Data/scene/Image.h>
#include <GVT/Data/primitives.h>
#include <GVT/Environment/RayTracerAttributes.h>

#include <pthread.h>
#include <semaphore.h>

#include <set>
using namespace std;

class RayTracer
{
 public:
    RayTracer() {}

    void RenderImage(string);
    GVT::Data::RayVector rays;


 protected:

    struct LoadBalancer
    {
      LoadBalancer(size_t size_, int granularity_=16)
        : size(size_), granularity(granularity_)
      {
        blockSize = max(size_t(1),size/granularity);
        last = 0;
      }
      void GetWork(size_t& begin, size_t& end)
      {
        begin = min(last, size);
        last += blockSize;
        end = last-1;
      }
      size_t size, blockSize, last;
      int granularity;
    };

    void IntersectQueueHandler(void* );
    std::vector<pthread_t> _threads;
    sem_t mutex;
    LoadBalancer* loadBalancer;

private:
	void parallel_comm_2(GVT::Env::RayTracerAttributes& rta, GVT::Data::RayVector& rays,
			Image& image);
};


#endif // GVT_RAY_TRACER_H

