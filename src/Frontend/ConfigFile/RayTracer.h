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

    void RenderImage(GVT::Env::RayTracerAttributes&, string);

 protected:
//    void                     MakeCameraRays(GVT::Data::RayVector&, GVT::Env::RayTracerAttributes::View&, float);

//    void                     TraverseDomain(Ray&, GVT::Data::RayVector&, GVT::Env::RayTracerAttributes&,
//                                            VolumeDomain*, float, const unsigned char*);
//    void                     TraverseQueue(std::map<int,GVT::Data::RayVector>&, GVT::Data::RayVector&, int, float,
//                                           const unsigned char*, GVT::Env::RayTracerAttributes&,
//                                           VolumeDomain*, ColorAccumulator*, long&, long&);
//    void                     IntersectDomain(Ray&, GVT::Data::RayVector&,
//                                             /* Carson TODO */ GeometryDomain*,
//                                             GVT::Env::RayTracerAttributes& );
//    void                     IntersectQueue(std::map<int,GVT::Data::RayVector>&, GVT::Data::RayVector&, int,
//                                            /* Carson TODO */ GeometryDomain*,
//                                            GVT::Env::RayTracerAttributes&, ColorAccumulator*,
//                                            long&, long&);

//    void                     ImageTrace( GVT::Env::RayTracerAttributes&, GVT::Data::RayVector&, Image& );
//    void                     DomainTrace( GVT::Env::RayTracerAttributes&, GVT::Data::RayVector&, Image& );
//    void                     HybridTrace( GVT::Env::RayTracerAttributes&, GVT::Data::RayVector&, Image& );


    //simple braindead load balancer
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

