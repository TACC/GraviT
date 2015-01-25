#ifndef GVT_CAMERA_H
#define GVT_CAMERA_H

#include <GVT/common/camera_config.h>
#include <GVT/Math/GVTMath.h>
#include <GVT/Data/primitives.h>
//#include <GVT/Environment/RayTracerAttributes.h>
#include <time.h>
#include <GVT/Concurrency/TaskScheduling.h>

#include <boost/foreach.hpp>
#include <boost/aligned_storage.hpp>

#include <boost/timer/timer.hpp>


namespace GVT {
    namespace Env {
#define PI 3.14159265359
#define SHOW(x) (cerr << #x << " = " << (x) << "\n")

        
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
            
            void SetCamera(GVT::Data::RayVector &rays, float rate);

            void setFilmSize(int width, int height) {
                
                filmsize[0] = width;
                filmsize[1] = height;
                
                setAspectRatio(double(width)/double(height));
                
            }
            
            float getFilmSizeWidth(void) {
                
                return filmsize[0];
                
            }
            
            float getFilmSizeHeight(void) {
                
                return filmsize[1];
                
            }
            
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
                eye = eyePos; look=lookAt; up = upDir;              
                GVT::Math::Vector3f z = -(lookAt - eyePos).normalize(); // this is where the z axis should end up
                const GVT::Math::Vector3f y = upDir; // where the y axis should end up
                GVT::Math::Vector3f x = (y ^ z).normalize(); // lah,
                m = GVT::Math::AffineTransformMatrix<float>(x[0], x[1], x[2], 0.f, y[0], y[1], y[2], 0.f, z[0], z[1], z[2], 0.f, 0.f, 0.f, 0.f, 1.f).transpose();
                update();
                const GVT::Math::AffineTransformMatrix<float> minv = m.inverse();
            }

            void setFOV(double fov) {
                normalizedHeight = tan(fov / 2.0);
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

            GVT::Data::RayVector& MakeCameraRays();

            boost::mutex rmutex;

//        private:
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
            GVT::Math::Vector4f up; // direction to look
            GVT::Math::Vector4f u, v; // u and v in the 

            GVT::Data::RayVector rays;
            float rate;
            int trcUpSampling;
            int depth;
            int filmsize[2];
        };
    }
}
#endif
