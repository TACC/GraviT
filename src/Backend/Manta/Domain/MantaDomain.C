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

namespace GVT {
    namespace Domain {

        MantaDomain::MantaDomain(string filename, GVT::Math::AffineTransformMatrix<float> m) : GVT::Domain::GeometryDomain(filename, m) {
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
            boost::atomic<int>& current_head;
            boost::atomic<int>& current_end;
            
            const size_t workSize = 4096;
            

            parallelTrace(
                    GVT::Domain::MantaDomain* dom,
                    GVT::Data::RayVector& rayList,
                    GVT::Data::RayVector& moved_rays,
                    boost::atomic<int>& current_head,
                    boost::atomic<int>& current_end) :

            dom(dom), rayList(rayList), moved_rays(moved_rays), current_head(current_head), current_end(current_end) {

            }

            void operator()() {
                const size_t maxPacketSize = 64;

                Manta::RenderContext& renderContext = *dom->rContext;

                GVT::Data::RayVector rayPacket;
                GVT::Data::RayVector localQueue;
                Manta::RayPacketData rpData;
                
                
                


                while (!rayList.empty()) {
                    GVT::Data::ray* ray = NULL;
                    size_t work = 0;
                    while ((ray = dom->pop(rayList)) && localQueue.size() < workSize) {
                        localQueue.push_back(ray);
                        work++;
                    }
                    GVT_DEBUG(DBG_ALWAYS,"Got " << localQueue.size() << " rays");
                    while (!localQueue.empty()) {
                        rayPacket.clear();
                        while (rayPacket.size() < 64 && !localQueue.empty()) {
                            rayPacket.push_back(localQueue.back());
                            localQueue.pop_back();
                        }


                        Manta::RayPacket mRays(rpData, Manta::RayPacket::UnknownShape, 0, rayPacket.size(), 0, Manta::RayPacket::NormalizedDirections);
                        for (int i = 0; i < rayPacket.size(); i++) {
                            mRays.setRay(i, GVT::Data::transform<GVT::Data::ray, Manta::Ray>(dom->toLocal(*rayPacket[i])));
                        }

                        mRays.resetHits();
                        dom->as->intersect(renderContext, mRays);
                        mRays.computeNormals<false>(renderContext);

                        for (int pindex = 0; pindex < rayPacket.size(); pindex++) {

                            if (mRays.wasHit(pindex)) {

                                if (rayPacket[pindex]->type == GVT::Data::ray::SHADOW) {
                                    delete rayPacket[pindex];
                                    continue;
                                }

                                rayPacket[pindex]->t = mRays.getMinT(pindex);
                                GVT::Math::Vector4f normal = dom->toWorld(GVT::Data::transform<Manta::Vector, GVT::Math::Vector4f>(mRays.getNormal(pindex)));

                                for (int lindex = 0; lindex < dom->lights.size(); lindex++) {
                                    GVT::Data::ray* ray = new GVT::Data::ray(*rayPacket[pindex]);
                                    ray->domains.clear();
                                    ray->type = GVT::Data::ray::SHADOW;
                                    ray->origin = ray->origin + ray->direction * ray->t;
                                    ray->setDirection(dom->lights[lindex]->position - ray->origin);
                                    GVT::Data::Color c = dom->mesh->mat->shade(ray, normal, dom->lights[lindex]);
                                    ray->color = COLOR_ACCUM(1.f, c[0], c[1], c[2], 1.f);

                                    //dom->push(rayList, ray);
                                    localQueue.push_back(ray);
                                }

                                int ndepth = rayPacket[pindex]->depth - 1;

                                float p = 1.f - (float(rand()) / RAND_MAX);

                                if (ndepth > 0 && ray->w > p) {
                                    GVT::Data::ray* ray = new GVT::Data::ray(*rayPacket[pindex]);
                                    ray->domains.clear();
                                    ray->type = GVT::Data::ray::SECUNDARY;
                                    ray->origin = ray->origin + ray->direction * ray->t;
                                    ray->setDirection(dom->mesh->mat->CosWeightedRandomHemisphereDirection2(normal).normalize());
                                    ray->w = ray->w * (ray->direction * normal);
                                    ray->depth = ndepth;
                                    //dom->push(rayList, ray);
                                    localQueue.push_back(ray);
                                }
                                delete rayPacket[pindex];
                                continue;
                            }
                            dom->dispatch(moved_rays, rayPacket[pindex]);
                        }
                    }
                }



#if 0

                GVT::Data::RayVector rayPacket;
                GVT::Data::RayVector rayPacket;
                Manta::RayPacketData rpData;
                //GVT_DEBUG(DBG_ALWAYS,"HERE 0");

                rayPacket.reserve(4096);
                rayPacket.reserve(64);

                while (current_head < current_end) {

                    if (current_head < current_end) {
                        boost::upgrade_lock<boost::shared_mutex> lock(dom->_inqueue);
                        boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
                        int popIDX;
                        for (popIDX = 0; popIDX < 1024 && !rayList.empty(); popIDX++) {
                            rayPacket.push_back(rayList[current_head + popIDX]);
                        }
                        current_head += popIDX;
                    }

                    while (!rayPacket.empty()) {
                        rayPacket.clear();
                        GVT::Data::ray* ray = NULL;
                        size_t psize = 0;
                        while (psize < 64 && !rayPacket.empty()) {
                            rayPacket.push_back(rayPacket.front());
                            psize++;

                        }

                        if (psize > 0) rayPacket.erase(rayPacket.begin(), rayPacket.begin() + psize);


                        Manta::RayPacket mRays(rpData, Manta::RayPacket::UnknownShape, 0, rayPacket.size(), 0, Manta::RayPacket::NormalizedDirections);

                        for (int i = 0; i < rayPacket.size(); i++) {
                            mRays.setRay(i, GVT::Data::transform<GVT::Data::ray, Manta::Ray>(dom->toLocal(*rayPacket[i])));
                        }

                        mRays.resetHits();
                        dom->as->intersect(*(dom->rContext), mRays);
                        mRays.computeNormals<false>(*(dom->rContext));

                        for (int pindex = 0; pindex < rayPacket.size(); pindex++) {

                            if (mRays.wasHit(pindex)) {

                                if (rayPacket[pindex]->type == GVT::Data::ray::SHADOW) {
                                    delete rayPacket[pindex];
                                    continue;
                                }

                                rayPacket[pindex]->t = mRays.getMinT(pindex);
                                GVT::Math::Vector4f normal = dom->toWorld(GVT::Data::transform<Manta::Vector, GVT::Math::Vector4f>(mRays.getNormal(pindex)));

                                for (int lindex = 0; lindex < dom->lights.size(); lindex++) {
                                    GVT::Data::ray* ray = new GVT::Data::ray(*rayPacket[pindex]);
                                    ray->domains.clear();
                                    ray->type = GVT::Data::ray::SHADOW;
                                    ray->origin = ray->origin + ray->direction * ray->t;
                                    ray->setDirection(dom->lights[lindex]->position - ray->origin);
                                    GVT::Data::Color c = dom->mesh->mat->shade(ray, normal, dom->lights[lindex]);
                                    ray->color = COLOR_ACCUM(1.f, c[0], c[1], c[2], 1.f);
                                    rayPacket.push_back(ray);
                                    //dom->push(rayList, ray);
                                    //current_end++;

                                }

                                int ndepth = rayPacket[pindex]->depth - 1;

                                float p = 1.f - (float(rand()) / RAND_MAX);

                                if (ndepth > 0 && ray->w > p) {
                                    GVT::Data::ray* ray = new GVT::Data::ray(*rayPacket[pindex]);
                                    ray->domains.clear();
                                    ray->type = GVT::Data::ray::SECUNDARY;
                                    ray->origin = ray->origin + ray->direction * ray->t;
                                    ray->setDirection(dom->mesh->mat->CosWeightedRandomHemisphereDirection2(normal).normalize());
                                    ray->w = ray->w * (ray->direction * normal);
                                    ray->depth = ndepth;
                                    rayPacket.push_back(ray);
                                    //dom->push(rayList, ray);
                                    //current_end++;
                                }
                                delete rayPacket[pindex];
                                continue;
                            }
                            dom->dispatch(moved_rays, rayPacket[pindex]);
                        }
                    }
                }
#endif
            }


        };

        void MantaDomain::trace(GVT::Data::RayVector& rayList, GVT::Data::RayVector& moved_rays) {
            GVT_DEBUG(DBG_ALWAYS, "processQueue<MantaDomain>: " << rayList.size());
            GVT_DEBUG(DBG_ALWAYS, "tracing geometry of domain " << domainID);
            boost::atomic<int> current_head(0);
            boost::atomic<int> current_end(rayList.size());

            for (int rc = 0; rc < GVT::Concurrency::asyncExec::instance()->numThreads; ++rc) {
                GVT::Concurrency::asyncExec::instance()->run_task(parallelTrace(this, rayList, moved_rays, current_head, current_end));
            }
            GVT::Concurrency::asyncExec::instance()->sync();



            GVT_DEBUG(DBG_ALWAYS, "Proccessed rays : " << rayList.size());
            GVT_DEBUG(DBG_ALWAYS, "Forwarding rays : " << moved_rays.size());
            rayList.clear();
#if 0
            const size_t maxPacketSize = 64;


            GVT::Data::RayVector localQueue;
            Manta::RayPacketData rpData;


            while (!rayList.empty()) {
                size_t psize = std::min(maxPacketSize, rayList.size());
                localQueue.clear();
                GVT::Data::ray* ray = NULL;

                for (int i = 0; i < psize; i++)
                    if ((ray = pop(rayList))) {
                        localQueue.push_back(ray);
                    }

                Manta::RayPacket mRays(rpData, Manta::RayPacket::UnknownShape, 0, localQueue.size(), 0, Manta::RayPacket::NormalizedDirections);
                for (int i = 0; i < localQueue.size(); i++) {
                    mRays.setRay(i, GVT::Data::transform<GVT::Data::ray, Manta::Ray>(toLocal(*localQueue[i])));
                }

                mRays.resetHits();
                as->intersect(*rContext, mRays);
                mRays.computeNormals<false>(*rContext);

                for (int pindex = 0; pindex < localQueue.size(); pindex++) {

                    if (mRays.wasHit(pindex)) {

                        if (localQueue[pindex]->type == GVT::Data::ray::SHADOW) {
                            delete localQueue[pindex];
                            continue;
                        }

                        localQueue[pindex]->t = mRays.getMinT(pindex);
                        GVT::Math::Vector4f normal = toWorld(GVT::Data::transform<Manta::Vector, GVT::Math::Vector4f>(mRays.getNormal(pindex)));

                        for (int lindex = 0; lindex < lights.size(); lindex++) {
                            GVT::Data::ray* ray = new GVT::Data::ray(*localQueue[pindex]);
                            ray->domains.clear();
                            ray->type = GVT::Data::ray::SHADOW;
                            ray->origin = ray->origin + ray->direction * ray->t;
                            ray->setDirection(lights[lindex]->position - ray->origin);
                            GVT::Data::Color c = mesh->mat->shade(ray, normal, lights[lindex]);
                            ray->color = COLOR_ACCUM(1.f, c[0], c[1], c[2], 1.f);

                            push(rayList, ray);

                        }

                        int ndepth = localQueue[pindex]->depth - 1;

                        float p = 1.f - (float(rand()) / RAND_MAX);

                        if (ndepth > 0 && ray->w > p) {
                            GVT::Data::ray* ray = new GVT::Data::ray(*localQueue[pindex]);
                            ray->domains.clear();
                            ray->type = GVT::Data::ray::SECUNDARY;
                            ray->origin = ray->origin + ray->direction * ray->t;
                            ray->setDirection(mesh->mat->CosWeightedRandomHemisphereDirection2(normal).normalize());
                            ray->w = ray->w * (ray->direction * normal);
                            ray->depth = ndepth;
                            push(rayList, ray);
                        }
                        delete localQueue[pindex];
                        continue;
                    }
                    dispatch(moved_rays, localQueue[pindex]);
                }
            }
            GVT_DEBUG(DBG_ALWAYS, "Done. Forwarding rays : " << moved_rays.size());
#endif
        }
    };
};


