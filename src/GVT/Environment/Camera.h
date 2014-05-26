#ifndef GVT_CAMERA_H
#define GVT_CAMERA_H

#include <GVT/common/camera_config.h>
#include <GVT/Math/GVTMath.h>
#include <GVT/Data/primitives.h>
#include <GVT/Environment/RayTracerAttributes.h>

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
                return ((float) rand() / RAND_MAX) - 0.5f * 2.0f;
            }

            double gauss(double x) {

                return 0.5 * exp(-((x - 1.0)*(x - 1.0)) / 0.2);

            }

            void MakeCameraRays() {
                int trcUpSampling = 1;
                rays.reserve( (trcUpSampling * trcUpSampling) * vi.width * vi.height );
                
                double divider = trcUpSampling;
                double offset = 1.0 / divider;
                double idivider2 = 1.0 / (divider * divider);
                double buffer_width = vi.width;
                double buffer_height = vi.height;
                GVT::Math::Vector4f dir;
                for (int j = 0; j < buffer_height; j++) {
                    for (int i = 0; i < buffer_width; i++) {
                        int idx = j * vi.width + i;
                        for (double off_i = 0; off_i < 1.0; off_i += offset) {
                            for (double off_j = 0; off_j < 1.0; off_j += offset) {
                                double x1 = double(i) + off_i + (offset / 2.0) * (frand() - 0.5);
                                double y1 = double(j) + off_j + (offset / 2.0) * (frand() - 0.5);
                                double x = x1 / double(buffer_width) - 0.5;
                                double y = y1 / double(buffer_height) - 0.5;
                                dir = m * ((look + x * u + y * v)).normalize();
                                GVT::Data::ray r(eye, dir, idivider2, GVT::Data::ray::PRIMARY, 2);
                                r.id = idx;
                                rays.push_back(r);
                            }
                        }
                    }
                }
                GVT_DEBUG(DBG_ALWAYS,"PREGENERATING : " << rays.size());
                
            }

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
