#include <gvt/render/data/scene/Camera.h>

#include <gvt/core/Math.h>

using namespace gvt::core::math;
using namespace gvt::render::actor;
using namespace gvt::render::data::scene;
using namespace std;

void Camera::SetCamera(RayVector &rays, float rate) 
{           
    this->rays = rays;
    this->rate = rate;
    this->trcUpSampling = 1;
}     

struct cameraGenerateRays 
{
    Camera* cam;
    size_t start, end;

    cameraGenerateRays(Camera* cam, size_t start, size_t end) 
    : cam(cam), start(start), end(end) 
    {}

    inline float frand() 
    {
        return ((float) rand() / RAND_MAX) - 0.5f * 2.0f;
    }

    void operator()() 
    {
        AffineTransformMatrix<float> m = cam->m; // rotation matrix
        int depth = cam->depth;
        RayVector& rays = cam->rays;
        Vector4f eye = cam->eye;
        Vector4f look = Vector4f(0,0,-1,0);//cam->look; // direction to look
        Vector4f u = cam->u, v = cam->v; // u and v in the 
//                    int samples = (cam->trcUpSampling * cam->trcUpSampling);
//
//                    GVT::Data::RayVector lrays;

//                    const GVT::Math::Vector4f u = GVT::Math::Vector4f(1, 0, 0, 0);
//                    const GVT::Math::Vector4f v = GVT::Math::Vector4f(0, 1, 0, 0);

        const float divider = cam->trcUpSampling;
        const float offset = 1.0 / divider;
        const float offset2 = offset / 2.f;
        const float w = 1.0 / (divider * divider);
        const float buffer_width = cam->getFilmSizeWidth();
        const float buffer_height = cam->getFilmSizeHeight();
        Vector4f dir;
        for (int j = start; j < end; j++) 
        {
            for (int i = 0; i < buffer_width; i++) 
            {
                int idx = j * buffer_width + i;
                for (float off_i = 0; off_i < 1.0; off_i += offset) 
                {
                    for (float off_j = 0; off_j < 1.0; off_j += offset) 
                    {
                        float x1 = float(i) + off_i + offset2 * (frand() - 0.5);
                        float y1 = float(j) + off_j + offset2 * (frand() - 0.5);
                        float x = x1 / float(buffer_width) - 0.5;
                        float y = y1 / float(buffer_height) - 0.5;

                        dir = m * ((look + x * u + y * v)).normalize();

                        Ray& ray = rays[idx];
                        ray.id = idx;;
                        ray.origin = eye;
                        ray.w = w;
                        ray.depth =  depth;
                        ray.setDirection(dir);
                        ray.type = Ray::PRIMARY;
                    }
                }
            }
        }
    }
};


RayVector& Camera::MakeCameraRays() 
{
    trcUpSampling = 1;
    depth = 0;
    size_t nrays = (trcUpSampling * trcUpSampling) * filmsize[0] * filmsize[1];
    // rays.reserve(nrays);
    int offset = filmsize[1] / gvt::core::schedule::asyncExec::instance()->numThreads;
    {
        boost::timer::auto_cpu_timer t("Allocate camera rays %t\n");
        rays.resize(nrays);
        // for(int i = 0; i< nrays; i++) {
        //     rays.push_back(GVT::Data::ray());
        // }
    }
    
    {
        boost::timer::auto_cpu_timer t("Generating camera rays %t\n");
        cameraGenerateRays(this, 0, filmsize[1])();
        // for (int start = 0; start < filmsize[1];) {
        //     int end = start + offset;
        //     end = std::min(end, filmsize[1]);
        //     GVT::Concurrency::asyncExec::instance()->run_task(cameraGenerateRays(this, start, end));
        //     start = end;
        // }
        // GVT::Concurrency::asyncExec::instance()->sync();
    }

    GVT_DEBUG(DBG_ALWAYS, "EXPECTED PREGENERATING : " << nrays);
    GVT_DEBUG(DBG_ALWAYS, "PREGENERATING : " << rays.size());
    return rays;
}


