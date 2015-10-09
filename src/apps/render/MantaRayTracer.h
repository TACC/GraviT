//
//  RayTracer.h
//


#ifndef GVTAPPS_RENDER_MANTA_RAY_TRACER_H
#define GVTAPPS_RENDER_MANTA_RAY_TRACER_H

#include <gvt/render/RenderContext.h>
#include <gvt/render/data/Domains.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/Image.h>
#include <apps/render/ConfigFileLoader.h>

#include <pthread.h>
#include <semaphore.h>

#include <algorithm>
#include <set>

namespace gvtapps {
    namespace render {

        class MantaRayTracer
        {
        public:
            MantaRayTracer(gvtapps::render::ConfigFileLoader& cl);

            void RenderImage(std::string);
            gvt::render::actor::RayVector rays;
            gvt::render::data::Dataset *scene;
			gvt::core::CoreContext *cntxt;
			gvt::core::DBNodeH root;



        protected:
            struct LoadBalancer
            {
                LoadBalancer(size_t size_, int granularity_=16)
                : size(size_), granularity(granularity_)
                {
                    blockSize = std::max(size_t(1),size/granularity);
                    last = 0;
                }

                void GetWork(size_t& begin, size_t& end)
                {
                    begin = std::min(last, size);
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
        };
    }
}

#endif // GVTAPPS_RENDER_MANTA_RAY_TRACER_H

