//
// MantaDomain.C
//

#include <Model/Primitives/Cube.h>
#include <Model/AmbientLights/AmbientOcclusionBackground.h>
#include <Model/AmbientLights/ArcAmbient.h>
#include <Model/AmbientLights/ConstantAmbient.h>
#include <Model/AmbientLights/EyeAmbient.h>
#include <Engine/Shadows/HardShadows.h>
#include <Engine/Shadows/NoShadows.h>
#include <Core/Math/CheapRNG.h>
#include <Core/Math/MT_RNG.h>

#include <GVT/common/utils.h>
#include <Domain/MantaDomain.h>
#include <Data/gvt_manta.h>
namespace GVT {
    namespace Domain {
        
        MantaDomain::MantaDomain(string filename, GVT::Math::AffineTransformMatrix<float> m) : GVT::Domain::GeometryDomain(filename,m) {
        }

        MantaDomain::MantaDomain(const MantaDomain& other) : GVT::Domain::GeometryDomain(other) {
        }
        
        MantaDomain::~MantaDomain() {
                //GeometryDomain::~GeometryDomain();
            
        }
        
        bool MantaDomain::load() {
            if(domainIsLoaded()) return true;
            
            GVT::Domain::GeometryDomain::load();
            //if(this->isLoaded) return true;
            //Manta::Mesh* mesh = new Manta::Mesh();
            Manta::Mesh* mesh = GVT::Data::transform<GVT::Data::Mesh*, Manta::Mesh*>(this->mesh);


            Manta::Material *material = new Manta::Lambertian(Manta::Color(Manta::RGBColor(0.f, 0.f, 0.f)));
            Manta::MeshTriangle::TriangleType triangleType = Manta::MeshTriangle::KENSLER_SHIRLEY_TRI;
            //string filename(filename);
            //readPlyFile(gdom->filename, Manta::AffineTransform::createIdentity(), mesh, material, triangleType);


            as = new Manta::DynBVH();
            as->setGroup(mesh);

            static Manta::MantaInterface* rtrt = Manta::createManta();
            Manta::LightSet* lights = new Manta::LightSet();
            lights->add(new Manta::PointLight(Manta::Vector(0, -5, 8), Manta::Color(Manta::RGBColor(1, 1, 1))));
            Manta::AmbientLight* ambient;
            ambient = new Manta::AmbientOcclusionBackground(Manta::Color::white()*0.5, 1, 36);
            Manta::Vector lightPosition(-10, 6, -30);
            Manta::PreprocessContext context(rtrt, 0, 1, lights);
            std::cout << "context.global_lights : " << context.globalLights << std::endl;
            material->preprocess(context);
            as->preprocess(context);
            Manta::ShadowAlgorithm* shadows;
            shadows = new Manta::HardShadows();
            Manta::Scene* scene = new Manta::Scene();


            scene->setLights(lights);
            scene->setObject(as);
            Manta::RandomNumberGenerator* rng = NULL;
            Manta::CheapRNG::create(rng);

            rContext = new Manta::RenderContext(rtrt, 0, 0/*proc*/, 1/*workersAnimandImage*/,
                    0/*animframestate*/,
                    0/*loadbalancer*/, 0/*pixelsampler*/, 0/*renderer*/, shadows/*shadowAlgorithm*/, 0/*camera*/, scene/*scene*/, 0/*thread_storage*/, rng/*rngs*/, 0/*samplegenerator*/);
            return true;
        }
        
        
        
        
        
    };
};


