/*
 * IntersectQueue.cpp
 *
 *  Created on: Nov 27, 2013
 *      Author: jbarbosa
 */



#include <GVT/Domain/domains.h>
#include <GVT/common/debug.h>

#include "MantaProcessQueue.h"

#include <Model/Primitives/Cube.h>
#include <Model/AmbientLights/AmbientOcclusionBackground.h>
#include <Model/AmbientLights/ArcAmbient.h>
#include <Model/AmbientLights/ConstantAmbient.h>
#include <Model/AmbientLights/EyeAmbient.h>
#include <Engine/Shadows/HardShadows.h>
#include <Engine/Shadows/NoShadows.h>
#include <Core/Math/CheapRNG.h>
#include <Core/Math/MT_RNG.h>
#include <queue>

#include <Data/gvt_manta.h>

#include <boost/timer/timer.hpp>

namespace GVT {
    namespace Backend {

        
        
        
        
        
        
        
        template<> void ProcessQueue<GVT::Domain::MantaDomain>::IntersectDomain(GVT::Data::ray& ray, GVT::Data::RayVector& newRays) {

        }

        template<> void ProcessQueue<GVT::Domain::MantaDomain>::operator()() {
#if 0
            GVT::Domain::MantaDomain* gdom = dynamic_cast<GVT::Domain::MantaDomain*> (param->dom);
            if (!gdom) return;
<<<<<<< HEAD
            boost::timer::auto_cpu_timer t;
            //GVT::Domain::MantaDomain* mdom = dynamic_cast<GVT::Domain::MantaDomain*> (param->dom);
            GVT::Data::RayVector& rayList = param->queue[param->domTarget];
            GVT_DEBUG(DBG_ALWAYS, "processQueue<MantaDomain>: " << rayList.size());

            //Manta::Mesh* mesh = new Manta::Mesh();
            Manta::Mesh* mesh = GVT::Data::transform<GVT::Data::Mesh*, Manta::Mesh*>(param->dom->mesh);

            Manta::Material *material = new Manta::Lambertian(Manta::Color(Manta::RGBColor(0.f, 0.f, 0.f)));
            Manta::MeshTriangle::TriangleType triangleType = Manta::MeshTriangle::KENSLER_SHIRLEY_TRI;
            //string filename(filename);
            //readPlyFile(gdom->filename, Manta::AffineTransform::createIdentity(), mesh, material, triangleType);

            Manta::DynBVH* as = new Manta::DynBVH();
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

            GVT::Data::LightSource* light = gdom->lights[0];
            GVT::Data::Material* mat = gdom->mesh->mat;
            // GVT::Data::Material* mat = material;

            Manta::RenderContext* rContext = new Manta::RenderContext(rtrt, 0, 0/*proc*/, 1/*workersAnimandImage*/,
                    0/*animframestate*/,
                    0/*loadbalancer*/, 0/*pixelsampler*/, 0/*renderer*/, shadows/*shadowAlgorithm*/, 0/*camera*/, scene/*scene*/, 0/*thread_storage*/, rng/*rngs*/, 0/*samplegenerator*/);
            Manta::RenderContext& renderContext = *(rContext);
=======
            GVT::Data::RayVector& rayList = param->queue[param->domTarget];
            GVT_DEBUG(DBG_ALWAYS, "processQueue<MantaDomain>: " << rayList.size());

            GVT::Data::Material* mat = gdom->mesh->mat;
            Manta::RenderContext& renderContext = *(gdom->rContext);
>>>>>>> 9195a9dd4e51e1563d6de48798fed773f7c3ffe5

            GVT_DEBUG(DBG_ALWAYS, "tracing geometry of domain " << param->domTarget);

            const size_t maxPacketSize = 64;


            std::vector<GVT::Data::ray*> localQueue;
            Manta::RayPacketData rpData;

            
            while (!rayList.empty()) {
                size_t psize = std::min(maxPacketSize, rayList.size());
                localQueue.clear();
                GVT::Data::ray* ray = NULL;

                for (int i = 0; i < psize; i++)
                    if ( (ray = pop(rayList))) {
                        localQueue.push_back(ray);
<<<<<<< HEAD

=======
>>>>>>> 9195a9dd4e51e1563d6de48798fed773f7c3ffe5
                    }

                Manta::RayPacket mRays(rpData, Manta::RayPacket::UnknownShape, 0, localQueue.size(), 0, Manta::RayPacket::NormalizedDirections);
                for (int i = 0; i < localQueue.size(); i++) {
                    mRays.setRay(i, GVT::Data::transform<GVT::Data::ray, Manta::Ray>(gdom->toLocal(*localQueue[i])));
                }

                mRays.resetHits();
                gdom->as->intersect(renderContext, mRays);
                mRays.computeNormals<false>(renderContext);

                for (int pindex = 0; pindex < localQueue.size(); pindex++) {

                    if (mRays.wasHit(pindex)) {
<<<<<<< HEAD

                        if (localQueue[pindex].type == GVT::Data::ray::SHADOW) continue;

                        localQueue[pindex].t = mRays.getMinT(pindex);
                        GVT::Math::Vector4f normal = GVT::Data::transform<Manta::Vector, GVT::Math::Vector4f>(mRays.getNormal(pindex));

                        for (int lindex = 0; lindex < gdom->lights.size(); lindex++) {
                            GVT::Data::ray ray(localQueue[pindex]);
                            ray.id = localQueue[pindex].id;
                            ray.origin_domain = param->domTarget;
                            ray.type = GVT::Data::ray::SHADOW;
                            ray.origin = ray.origin + ray.direction * ray.t;
                            ray.setDirection(gdom->lights[lindex]->position - ray.origin);

                            GVT::Data::Color c = mat->shade(ray, normal, gdom->lights[lindex]);

                            ray.color = COLOR_ACCUM(1.f, c[0], c[1], c[2], 1.f);

=======
                        
                        if (localQueue[pindex]->type == GVT::Data::ray::SHADOW) {
                            delete localQueue[pindex];
                            continue;
                        }
                        
                        localQueue[pindex]->t = mRays.getMinT(pindex);
                        GVT::Math::Vector4f normal = gdom->toWorld(GVT::Data::transform<Manta::Vector, GVT::Math::Vector4f>(mRays.getNormal(pindex)));
                        
                        for (int lindex = 0; lindex < gdom->lights.size(); lindex++) {
                            GVT::Data::ray* ray = new GVT::Data::ray(*localQueue[pindex]);
                            ray->domains.clear();
                            ray->type = GVT::Data::ray::SHADOW;
                            ray->origin = ray->origin + ray->direction * ray->t;
                            ray->setDirection(gdom->lights[lindex]->position - ray->origin);
                            GVT::Data::Color c = mat->shade(ray, normal, gdom->lights[lindex]);
                            ray->color = COLOR_ACCUM(1.f, c[0], c[1], c[2], 1.f);
                            
>>>>>>> 9195a9dd4e51e1563d6de48798fed773f7c3ffe5
                            push(rayList, ray);
  
                        }

                        int ndepth = localQueue[pindex]->depth - 1;
                        
                        float p = 1.f - (float(rand()) / RAND_MAX);
                        
                        if (ndepth > 0 && ray->w > p) {     
                            GVT::Data::ray* ray = new GVT::Data::ray(*localQueue[pindex]);
                            ray->domains.clear();
                            ray->type = GVT::Data::ray::SECUNDARY;
                            ray->origin = ray->origin + ray->direction * ray->t;
                            ray->setDirection(mat->CosWeightedRandomHemisphereDirection2(normal).normalize());
                            ray->w = ray->w * (ray->direction * normal);
                            ray->depth = ndepth;
                            push(rayList, ray);
                        }
                        delete localQueue[pindex];
                        continue;
                    }
                    dispatch(param->moved_rays, localQueue[pindex]);
                }
            }
            GVT_DEBUG(DBG_ALWAYS, "Done. Forwarding rays : " << param->moved_rays.size());
#endif
        }
    };
};
