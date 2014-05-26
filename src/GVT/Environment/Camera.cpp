#include <GVT/Math/GVTMath.h>
#include "Camera.h"


namespace GVT {
    namespace Env {


#define PI 3.14159265359
#define SHOW(x) (cerr << #x << " = " << (x) << "\n")

        using namespace std;

        template<> Camera<C_ORTHOGRAPHIC>::Camera(GVT::Data::RayVector &rays, GVT::Env::RayTracerAttributes::View &vi, float rate) : rays(rays), vi(vi), rate(rate), m(true) {
            if (vi.nearPlane == 0) vi.nearPlane = -vi.camera.z;
            if (vi.farPlane == 0) vi.farPlane = vi.camera.z;
            aspectRatio = 1;
            normalizedHeight = 1;
            eye = vi.camera;
            u = GVT::Math::Vector4f(1, 0, 0, 0);
            v = GVT::Math::Vector4f(0, 1, 0, 0);
            look = vi.camera - vi.focus;
            setLook(look, vi.up);
            setAspectRatio(double(vi.width)/double(vi.height));
        }

        template<> Camera<C_PERSPECTIVE>::Camera(GVT::Data::RayVector &rays, GVT::Env::RayTracerAttributes::View &vi, float rate) : rays(rays), vi(vi), rate(rate), m(true) {
            aspectRatio = 1;
            normalizedHeight = 1;
            eye = vi.camera;
       
            u = GVT::Math::Vector4f(1, 0, 0, 0);
            v = GVT::Math::Vector4f(0, 1, 0, 0);
           
            setLook(vi.camera, vi.focus , vi.up);
            setFOV(float(vi.view_angle) * PI / 180.f);
            setAspectRatio(double(vi.width)/double(vi.height));
            
        }
    }
}



