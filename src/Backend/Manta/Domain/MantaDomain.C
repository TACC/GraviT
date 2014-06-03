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

#include <boost/atomic.hpp>
#include <GVT/Concurrency/TaskScheduling.h>

#include "Environment/RayTracerAttributes.h"

static boost::atomic<size_t> counter(0);


namespace GVT {
    namespace Domain {

        MantaDomain::MantaDomain(string filename, GVT::Math::AffineTransformMatrix<float> m) : GVT::Domain::GeometryDomain(filename, m) {

            if (domainIsLoaded()) {
                Manta::Mesh* mesh = GVT::Data::transform<GVT::Data::Mesh*, Manta::Mesh*>(this->mesh);


                Manta::Material *material = new Manta::Lambertian(Manta::Color(Manta::RGBColor(0.f, 0.f, 0.f)));
                Manta::MeshTriangle::TriangleType triangleType = Manta::MeshTriangle::KENSLER_SHIRLEY_TRI;
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

            }


        }

        MantaDomain::MantaDomain(const MantaDomain& other) : GVT::Domain::GeometryDomain(other) {
        }

        MantaDomain::~MantaDomain() {
            //GeometryDomain::~GeometryDomain();

        }

        bool MantaDomain::load() {
            if (domainIsLoaded()) return true;

            GVT::Domain::GeometryDomain::load();
            Manta::Mesh* mesh = GVT::Data::transform<GVT::Data::Mesh*, Manta::Mesh*>(this->mesh);


            Manta::Material *material = new Manta::Lambertian(Manta::Color(Manta::RGBColor(0.f, 0.f, 0.f)));
            Manta::MeshTriangle::TriangleType triangleType = Manta::MeshTriangle::KENSLER_SHIRLEY_TRI;
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

        void MantaDomain::free() {
            return;
        }

        struct parallelTrace {
            GVT::Domain::MantaDomain* dom;
            GVT::Data::RayVector& rayList;
            GVT::Data::RayVector& moved_rays;
            const size_t workSize;
            
            boost::atomic<size_t>& counter;

            parallelTrace(
                    GVT::Domain::MantaDomain* dom,
                    GVT::Data::RayVector& rayList,
                    GVT::Data::RayVector& moved_rays,
                    const size_t workSize,
                    boost::atomic<size_t>& counter) :

            dom(dom), rayList(rayList), moved_rays(moved_rays), workSize(workSize), counter(counter) {
            }

            void operator()() {
                const size_t maxPacketSize = 64;

                Manta::RenderContext& renderContext = *dom->rContext;

                GVT::Data::RayVector rayPacket;
                GVT::Data::RayVector localQueue;
                GVT::Data::RayVector localDispatch;

                Manta::RayPacketData rpData;

                localQueue.reserve(workSize * 2);
                localDispatch.reserve(rayList.size() * 2);


                while (!rayList.empty()) {
                    boost::unique_lock<boost::mutex> queue(dom->_inqueue);
                    std::size_t range = std::min(workSize, rayList.size());
                    localQueue.assign(rayList.begin(), rayList.begin() + range);
                    rayList.erase(rayList.begin(), rayList.begin() + range);
                    queue.unlock();


                    GVT_DEBUG(DBG_ALWAYS, "Got " << localQueue.size() << " rays");
                    while (!localQueue.empty()) {
                        rayPacket.clear();

                        while (rayPacket.size() < 64 && !localQueue.empty()) {
                            rayPacket.push_back(localQueue.back());
                            localQueue.pop_back();
                        }


                        Manta::RayPacket mRays(rpData, Manta::RayPacket::UnknownShape, 0, rayPacket.size(), 0, Manta::RayPacket::NormalizedDirections);
                        for (int i = 0; i < rayPacket.size(); i++) {
                            mRays.setRay(i, GVT::Data::transform<GVT::Data::ray, Manta::Ray>(dom->toLocal(rayPacket[i])));
                        }

                        mRays.resetHits();
                        dom->as->intersect(renderContext, mRays);
                        mRays.computeNormals<false>(renderContext);

                        for (int pindex = 0; pindex < rayPacket.size(); pindex++) {

                            if (mRays.wasHit(pindex)) {

                                if (rayPacket[pindex].type == GVT::Data::ray::SHADOW) {
                                    counter++;
                                    continue;
                                }

                                float t = mRays.getMinT(pindex);
                                rayPacket[pindex].t = t;
                                GVT::Math::Vector4f normal = dom->toWorld(GVT::Data::transform<Manta::Vector, GVT::Math::Vector4f>(mRays.getNormal(pindex)));
                                if(rayPacket[pindex].type == GVT::Data::ray::SECUNDARY) {
                                    t = (t > 1) ? 1.f/t : t;
                                    rayPacket[pindex].w = rayPacket[pindex].w * t;
                                }

                                for (int lindex = 0; lindex < dom->lights.size(); lindex++) {
                                    GVT::Data::ray ray(rayPacket[pindex]);
                                    ray.domains.clear();
                                    ray.type = GVT::Data::ray::SHADOW;
                                    ray.origin = ray.origin + ray.direction * ray.t;
                                    ray.setDirection(dom->lights[lindex]->position - ray.origin);
                                    GVT::Data::Color c = dom->mesh->mat->shade(ray, normal, dom->lights[lindex]);
                                    ray.color = COLOR_ACCUM(1.f, c[0], c[1], c[2], 1.f);
                                    localQueue.push_back(ray);
                                }
                                
                                int ndepth = rayPacket[pindex].depth - 1;

                                float p = 1.f - (float(rand()) / RAND_MAX);
                                
                                if (ndepth > 0 && rayPacket[pindex].w > p) {
                                        GVT::Data::ray ray(rayPacket[pindex]);
                                        ray.domains.clear();
                                        ray.type = GVT::Data::ray::SECUNDARY;
                                        ray.origin = ray.origin + ray.direction * ray.t;
                                        ray.setDirection(dom->mesh->mat->CosWeightedRandomHemisphereDirection2(normal).normalize());
                                        ray.w = ray.w * (ray.direction * normal);
                                        ray.depth = ndepth;
                                        localQueue.push_back(ray);
                                }
                                counter++;
                                continue;
                            }
                            counter++;
                            localDispatch.push_back(rayPacket[pindex]);
                        }
                    }
                }

                boost::unique_lock<boost::mutex> moved(dom->_outqueue);
                moved_rays.insert(moved_rays.begin(), localDispatch.begin(), localDispatch.end());
                moved.unlock();
            }
        };

        void MantaDomain::trace(GVT::Data::RayVector& rayList, GVT::Data::RayVector& moved_rays) {
            GVT_DEBUG(DBG_ALWAYS, "processQueue<MantaDomain>: " << rayList.size());
            GVT_DEBUG(DBG_ALWAYS, "tracing geometry of domain " << domainID);

            size_t workload = std::max((size_t) 1, (size_t) (rayList.size() / (GVT::Concurrency::asyncExec::instance()->numThreads * 4)));
            
            for (int rc = 0; rc < GVT::Concurrency::asyncExec::instance()->numThreads; ++rc) {
                GVT::Concurrency::asyncExec::instance()->run_task(parallelTrace(this, rayList, moved_rays, workload,counter));
            }
            GVT::Concurrency::asyncExec::instance()->sync();

#ifdef NDEBUG            
            std::cout << "Proccessed rays : " << counter << std::endl;
#else
            GVT_DEBUG(DBG_ALWAYS, "Proccessed rays : " << counter);
#endif
            GVT_DEBUG(DBG_ALWAYS, "Forwarding rays : " << moved_rays.size());
            rayList.clear();
        }
    };
};


