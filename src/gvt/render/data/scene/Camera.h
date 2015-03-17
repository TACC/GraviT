#ifndef GVT_RENDER_DATA_SCENE_CAMERA_H
#define GVT_RENDER_DATA_SCENE_CAMERA_H

#include <gvt/core/Math.h>
#include <gvt/core/schedule/TaskScheduling.h>
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/scene/CameraConfig.h>

#include <boost/aligned_storage.hpp>
#include <boost/foreach.hpp>
#include <boost/timer/timer.hpp>

#include <time.h>


namespace gvt {
    namespace render {
      namespace data {
        namespace scene {

                class Camera 
                {
                public:

                    Camera() 
                    {
                        aspectRatio = 1;
                        normalizedHeight = 1;
                        eye = gvt::core::math::Vector4f(0, 0, 0, 1);
                        u = gvt::core::math::Vector4f(1, 0, 0, 0);
                        v = gvt::core::math::Vector4f(0, 1, 0, 0);
                        look = gvt::core::math::Vector4f(0, 0, -1, 1);
                    }
                    
                    void SetCamera(gvt::render::actor::RayVector &rays, float rate);

                    void setFilmSize(int width, int height) 
                    {
                        filmsize[0] = width;
                        filmsize[1] = height;
                        
                        setAspectRatio(double(width)/double(height));
                    }
                    
                    float getFilmSizeWidth(void) 
                    {
                        return filmsize[0];
                    }
                    
                    float getFilmSizeHeight(void) 
                    {
                        return filmsize[1];
                    }
                    
                    void setEye(const gvt::core::math::Vector4f &eye) 
                    {
                        this->eye = eye;
                    }

                    void setLook(double r, double i, double j, double k) 
                    {
                        // set look matrix
                        m[0][0] = 1.0 - 2.0 * (i * i + j * j);
                        m[0][1] = 2.0 * (r * i - j * k);
                        m[0][2] = 2.0 * (j * r + i * k);

                        m[1][0] = 2.0 * (r * i + j * k);
                        m[1][1] = 1.0 - 2.0 * (j * j + r * r);
                        m[1][2] = 2.0 * (i * j - r * k);

                        m[2][0] = 2.0 * (j * r - i * k);
                        m[2][1] = 2.0 * (i * j + r * k);
                        m[2][2] = 1.0 - 2.0 * (i * i + r * r);

                        update();
                    }

                    void setLook(const gvt::core::math::Vector4f &viewDir, const gvt::core::math::Vector4f &upDir) 
                    {
                        gvt::core::math::Vector3f z = viewDir; // this is where the z axis should end up
                        const gvt::core::math::Vector3f &y = upDir; // where the y axis should end up
                        gvt::core::math::Vector3f x = y ^ z; // lah,
                        m = gvt::core::math::AffineTransformMatrix<float>(x[0], x[1], x[2], 0.f, y[0], y[1], y[2], 0.f, z[0], z[1], z[2], 0.f, 0.f, 0.f, 0.f, 1.f).transpose();
                        update();
                    }

                    void setLook(gvt::core::math::Vector4f &eyePos, gvt::core::math::Vector4f &lookAt, const gvt::core::math::Vector4f &upDir) 
                    {
                        eye = eyePos; look=lookAt; up = upDir;              
                        gvt::core::math::Vector3f z = -(lookAt - eyePos).normalize(); // this is where the z axis should end up
                        const gvt::core::math::Vector3f y = upDir; // where the y axis should end up
                        gvt::core::math::Vector3f x = (y ^ z).normalize(); // lah,
                        m = gvt::core::math::AffineTransformMatrix<float>(x[0], x[1], x[2], 0.f, y[0], y[1], y[2], 0.f, z[0], z[1], z[2], 0.f, 0.f, 0.f, 0.f, 1.f).transpose();
                        update();
                        const gvt::core::math::AffineTransformMatrix<float> minv = m.inverse();
                    }

                    void setFOV(double fov) 
                    {
                        normalizedHeight = tan(fov / 2.0);
                        update();
                    }

                    void setAspectRatio(double ar) 
                    {
                        aspectRatio = ar;
                        update();
                    }

                    double getAspectRatio() 
                    {
                        return aspectRatio;
                    }

                    const gvt::core::math::Vector4f& getEye() const 
                    {
                        return eye;
                    }

                    const gvt::core::math::Vector4f& getLook() const 
                    {
                        return look;
                    }

                    const gvt::core::math::Vector4f& getU() const 
                    {
                        return u;
                    }

                    const gvt::core::math::Vector4f& getV() const 
                    {
                        return v;
                    }

                    const gvt::core::math::AffineTransformMatrix<float> getMatrix() 
                    {
                        return m;
                    }

                    float frand() 
                    {
                        // srand(time(NULL));
                        return .1f;//((float) rand() / RAND_MAX) - 0.5f * 2.0f;
                    }

                    double gauss(double x) 
                    {
                        return 0.5 * exp(-((x - 1.0)*(x - 1.0)) / 0.2);
                    }

                    // void MakeCameraRays() {
                    //     trcUpSampling = 1;
                    //     depth = 0;
                    //     size_t nrays = (trcUpSampling * trcUpSampling) * vi.width * vi.height;
                    //     int offset = vi.height / GVT::Concurrency::asyncExec::instance()->numThreads;

                    //     {
                    //         boost::timer::auto_cpu_timer t("Allocate camera rays %t\n");
                    //                         rays.resize(nrays);
                    //         // for(int i = 0; i< nrays; i++) {
                    //             // rays.push_back(GVT::Data::ray());
                    //         // }
                    //     }
                        
                    //     {
                    //         boost::timer::auto_cpu_timer t("Generating camera rays %t\n");
                    //         cameraGenerateRays(this, 0, vi.height)();
                    //         // for (int start = 0; start < vi.height;) {
                    //         //     int end = start + offset;
                    //         //     end = std::min(end, vi.height);
                    //         //     GVT::Concurrency::asyncExec::instance()->run_task(cameraGenerateRays(this, start, end));
                    //         //     start = end;
                    //         // }
                    //         // GVT::Concurrency::asyncExec::instance()->sync();
                    //     }

                    //     GVT_DEBUG(DBG_ALWAYS, "EXPECTED PREGENERATING : " << (trcUpSampling * trcUpSampling) * vi.width * vi.height);
                    //     GVT_DEBUG(DBG_ALWAYS, "PREGENERATING : " << rays.size());

                    // }

                    // struct cameraGenerateRays {
                    //     Camera* cam;
                    //     size_t start, end;

                    //     cameraGenerateRays(Camera* cam, size_t start, size_t end) : cam(cam), start(start), end(end) {
                    //     }

                    //     inline float frand() {
                    //         return .1;//((float) rand() / RAND_MAX) - 0.5f * 2.0f;
                    //     }

                    //     void operator()() {
                    //         gvt::core::math::AffineTransformMatrix<float> m = cam->m; // rotation matrix
                    //         int depth = cam->depth;
                    //         GVT::Data::RayVector& rays = cam->rays;
                    //         gvt::core::math::Vector4f eye = cam->eye;
                    //         gvt::core::math::Vector4f look = cam->look; // direction to look
                    //         gvt::core::math::Vector4f u = cam->u, v = cam->v; // u and v in the 
                    //         int samples = (cam->trcUpSampling * cam->trcUpSampling);

                    //         GVT::Data::RayVector lrays;

                    //         const float divider = cam->trcUpSampling;
                    //         const float offset = 1.0 / divider;
                    //         const float offset2 = offset / 2.f;
                    //         const float w = 1.0 / (divider * divider);
                    //         const float buffer_width = cam->vi.width;
                    //         const float buffer_height = cam->vi.height;
                    //         gvt::core::math::Vector4f dir;
                    //         float x11 = 1.0f/float(buffer_width);
                    //         for (int j = start; j < end; j++) {
                    //             for (int i = 0; i < buffer_width; i++) {
                    //                 int idx = j * buffer_width + i;
                    //                 for (float off_i = 0; off_i < 1.0; off_i += offset) {
                    //                     for (float off_j = 0; off_j < 1.0; off_j += offset) {
                    //                         float x1 = float(i) + off_i + offset2 * (frand() - 0.5);
                    //                         float y1 = float(j) + off_j + offset2 * (frand() - 0.5);
                    //                         float x = x1 *x11 - 0.5;
                    //                         float y = y1 / float(buffer_height) - 0.5;
                    //                         dir = m * ((look + x * u + y * v)).normalize();
                    //                         GVT::Data::ray& ray = rays[idx];
                    //                         ray.id = idx;;
                    //                         ray.origin = eye;
                    //                         ray.w = w;
                    //                         ray.depth =  depth;
                    //                         ray.setDirection(dir);
                    //                         ray.type = GVT::Data::ray::PRIMARY;
                    //                     }
                    //                 }
                    //             }
                    //         }
                    //     }
                    // };
                    gvt::render::actor::RayVector& MakeCameraRays();

                    boost::mutex rmutex;

        //        private:
                    gvt::core::math::AffineTransformMatrix<float> m; // rotation matrix
                    double normalizedHeight; // dimensions of image place at unit dist from eye
                    double aspectRatio;

                    void update() 
                    { 
                        // using the above three values calculate look,u,v
                        u = m * gvt::core::math::Vector3f(1, 0, 0) * normalizedHeight*aspectRatio;
                        v = m * gvt::core::math::Vector3f(0, 1, 0) * normalizedHeight;
                        look = gvt::core::math::Vector3f(0, 0, -1);
                    }

                    gvt::core::math::Vector4f eye;
                    gvt::core::math::Vector4f look; // direction to look
                    gvt::core::math::Vector4f up; // direction to look
                    gvt::core::math::Vector4f u, v; // u and v in the 

                    gvt::render::actor::RayVector rays;
                    float rate;
                    int trcUpSampling;
                    int depth;
                    int filmsize[2];
                };
            }
        }
    }
}
#endif // GVT_RENDER_DATA_SCENE_CAMERA_H
