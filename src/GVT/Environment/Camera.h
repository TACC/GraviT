#ifndef GVT_CAMERA_H
#define GVT_CAMERA_H

#include <GVT/common/camera_config.h>
#include <GVT/Math/GVTMath.h>
#include <GVT/Data/primitives.h>
#include <GVT/Environment/RayTracerAttributes.h>
#include <time.h>
#include <GVT/Concurrency/TaskScheduling.h>

#include <boost/foreach.hpp>
#include <boost/aligned_storage.hpp>

namespace GVT {
    namespace Env {
#define PI 3.14159265359
#define SHOW(x) (cerr << #x << " = " << (x) << "\n")

        template <CAMERA_TYPE CT>
        class Camera {
        public:

            Camera() {
                aspectRatio = 1;
                normalizedHeight = 1;
                eye = GVT::Math::Vector4f(0, 0, 0, 1);
                u = GVT::Math::Vector4f(1, 0, 0, 0);
                v = GVT::Math::Vector4f(0, 1, 0, 0);
                look = GVT::Math::Vector4f(0, 0, -1, 1);
            }
            Camera(GVT::Data::RayVector &rays, GVT::Env::RayTracerAttributes::View &vi, float rate);

            void setEye(const GVT::Math::Vector4f &eye) {
                this->eye = eye;
            }

            void setLook(double r, double i, double j, double k) {
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

            void setLook(const GVT::Math::Vector4f &viewDir, const GVT::Math::Vector4f &upDir) {
                GVT::Math::Vector3f z = viewDir; // this is where the z axis should end up
                const GVT::Math::Vector3f &y = upDir; // where the y axis should end up
                GVT::Math::Vector3f x = y ^ z; // lah,
                m = GVT::Math::AffineTransformMatrix<float>(x[0], x[1], x[2], 0.f, y[0], y[1], y[2], 0.f, z[0], z[1], z[2], 0.f, 0.f, 0.f, 0.f, 1.f).transpose();
                update();
            }

            void setLook(GVT::Math::Vector4f &eyePos, GVT::Math::Vector4f &lookAt, const GVT::Math::Vector4f &upDir) {
                GVT::Math::Vector3f z = -(lookAt - eyePos).normalize(); // this is where the z axis should end up
                const GVT::Math::Vector3f y = upDir; // where the y axis should end up
                GVT::Math::Vector3f x = (y ^ z).normalize(); // lah,
                this->eye = eyePos;
                m = GVT::Math::AffineTransformMatrix<float>(x[0], x[1], x[2], 0.f, y[0], y[1], y[2], 0.f, z[0], z[1], z[2], 0.f, 0.f, 0.f, 0.f, 1.f).transpose();
                update();
                const GVT::Math::AffineTransformMatrix<float> minv = m.inverse();
            }

            void setFOV(double fov) {
                normalizedHeight = 2 * tan(fov / 2);
                update();
            }

            void setAspectRatio(double ar) {
                aspectRatio = ar;
                update();
            }

            double getAspectRatio() {
                return aspectRatio;
            }

            const GVT::Math::Vector4f& getEye() const {
                return eye;
            }

            const GVT::Math::Vector4f& getLook() const {
                return look;
            }

            const GVT::Math::Vector4f& getU() const {
                return u;
            }

            const GVT::Math::Vector4f& getV() const {
                return v;
            }

            const GVT::Math::AffineTransformMatrix<float> getMatrix() {
                return m;
            }

            float frand() {
                srand(time(NULL));
                return ((float) rand() / RAND_MAX) - 0.5f * 2.0f;
            }

            double gauss(double x) {

                return 0.5 * exp(-((x - 1.0)*(x - 1.0)) / 0.2);

            }

            void MakeCameraRays() {
                trcUpSampling = 2;
                depth = 4;
                rays.reserve((trcUpSampling*trcUpSampling) * vi.width * vi.height);
                int offset = vi.height / GVT::Concurrency::asyncExec::instance()->numThreads;
                for (int start = 0; start < vi.height;) {
                    int end = start + offset;
                    end = std::min(end, vi.height);
                    GVT::Concurrency::asyncExec::instance()->run_task(cameraGenerateRays(this, start, end));
                    start = end;
                }
                GVT::Concurrency::asyncExec::instance()->sync();
                GVT_DEBUG(DBG_ALWAYS, "EXPECTED PREGENERATING : " << (trcUpSampling*trcUpSampling) * vi.width * vi.height);
                GVT_DEBUG(DBG_ALWAYS, "PREGENERATING : " << rays.size());

            }

            struct cameraGenerateRays {
                Camera* cam;
                size_t start, end;

                cameraGenerateRays(Camera* cam, size_t start, size_t end) : cam(cam), start(start), end(end) {
                }

                inline float frand() {
                    return ((float) rand() / RAND_MAX) - 0.5f * 2.0f;
                }

                void operator()() {
                    GVT::Math::AffineTransformMatrix<float> m = cam->m; // rotation matrix
                    int depth = cam->depth;
                    GVT::Data::RayVector& rays = cam->rays;
                    GVT::Math::Vector4f eye = cam->eye;
                    GVT::Math::Vector4f look = cam->look; // direction to look
                    GVT::Math::Vector4f u = cam->u, v = cam->v; // u and v in the 
                    int samples = (cam->trcUpSampling * cam->trcUpSampling);
                    
                    GVT::Data::RayVector lrays;
                    
                    const float divider = cam->trcUpSampling;
                    const float offset = 1.0 / divider;
                    const float offset2 = offset / 2.f;
                    const float idivider2 = 1.0 / (divider * divider);
                    const float buffer_width = cam->vi.width;
                    const float buffer_height = cam->vi.height;
                    GVT::Math::Vector4f dir;
                    for (int j = start; j < end; j++) {
                        for (int i = 0; i < buffer_width; i++) {
                            int idx = j * buffer_width + i;
                            for (float off_i = 0; off_i < 1.0; off_i += offset) {
                                for (float off_j = 0; off_j < 1.0; off_j += offset) {
                                    float x1 = float(i) + off_i + offset2 * (frand() - 0.5);
                                    float y1 = float(j) + off_j + offset2 * (frand() - 0.5);
                                    float x = x1 / float(buffer_width) - 0.5;
                                    float y = y1 / float(buffer_height) - 0.5;
                                    dir = m * ((look + x * u + y * v)).normalize();
                                    GVT::Data::ray* ray = new GVT::Data::ray(eye, dir, idivider2, GVT::Data::ray::PRIMARY, depth);
                                    ray->id = idx;
                                    lrays.push_back(ray);
                                }
                            }
                        }
                    }
                    boost::mutex::scoped_lock lock(cam->rmutex);
                    rays.insert(rays.end(), lrays.begin(), lrays.end());
                }
            };

            boost::mutex rmutex;
            
        private:
            GVT::Math::AffineTransformMatrix<float> m; // rotation matrix
            double normalizedHeight; // dimensions of image place at unit dist from eye
            double aspectRatio;

            void update() { // using the above three values calculate look,u,v
                u = m * GVT::Math::Vector3f(1, 0, 0) * normalizedHeight*aspectRatio;
                v = m * GVT::Math::Vector3f(0, 1, 0) * normalizedHeight;
                look = GVT::Math::Vector3f(0, 0, -1);
            }

            GVT::Math::Vector4f eye;
            GVT::Math::Vector4f look; // direction to look
            GVT::Math::Vector4f u, v; // u and v in the 

            GVT::Data::RayVector &rays;
            GVT::Env::RayTracerAttributes::View &vi;
            float rate;

            int trcUpSampling;
            int depth;
            
            
        };

        template<> Camera<C_ORTHOGRAPHIC>::Camera(GVT::Data::RayVector& rays, GVT::Env::RayTracerAttributes::View& vi, float rate);
        template<> Camera<C_PERSPECTIVE>::Camera(GVT::Data::RayVector& rays, GVT::Env::RayTracerAttributes::View& vi, float rate);
        //        template <CAMERA_TYPE CT> void Camera<CT>::rayThrough(double x, double y, GVT::Data::ray &r);
        //        template <CAMERA_TYPE CT> void Camera<CT>::setEye(const GVT::Math::Vector4f &eye);
        //        template <CAMERA_TYPE CT> void Camera<CT>::setLook(double, double, double, double);
        //        template <CAMERA_TYPE CT> void Camera<CT>::setLook(const GVT::Math::Vector4f &viewDir, const GVT::Math::Vector4f &upDir);
        //        template <CAMERA_TYPE CT> void Camera<CT>::setLook(GVT::Math::Vector4f &eyePos, GVT::Math::Vector4f &lookAt, const GVT::Math::Vector4f &upDir);
        //
        //        template <CAMERA_TYPE CT> void Camera<CT>::setFOV(double);
        //        template <CAMERA_TYPE CT> void Camera<CT>::setAspectRatio(double);
    }
}
#endif
