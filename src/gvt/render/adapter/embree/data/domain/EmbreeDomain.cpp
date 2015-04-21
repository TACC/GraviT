//
// EmbreeDomain.cpp
//

#include <gvt/render/adapter/embree/data/domain/EmbreeDomain.h>

#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>
#include <gvt/core/schedule/TaskScheduling.h>
#include <gvt/render/actor/Ray.h>
#include <gvt/render/adapter/embree/data/Transforms.h>
#include <gvt/render/Attributes.h>
#include <gvt/render/data/scene/ColorAccumulator.h>
#include <gvt/render/data/scene/Light.h>

// Embree includes
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
// end Embree includes

#include <boost/atomic.hpp>
#include <boost/foreach.hpp>

#define GVT_PACKET_SIZE 4

using namespace gvt::render::actor;
using namespace gvt::render::adapter::embree::data;
using namespace gvt::render::adapter::embree::data::domain;
using namespace gvt::render::data::domain;
using namespace gvt::render::data::primitives;
using namespace gvt::core::math;

static boost::atomic<size_t> counter(0);

bool EmbreeDomain::init = false;

EmbreeDomain::EmbreeDomain(GeometryDomain* domain) : GeometryDomain(*domain)
{
    GVT_DEBUG(DBG_ALWAYS, "Converting domain");

    if(!EmbreeDomain::init) {
        rtcInit(0);
        EmbreeDomain::init = true;
    }

    switch(GVT_PACKET_SIZE) {
        case 4: packetSize = RTC_INTERSECT4; break;
        case 8: packetSize = RTC_INTERSECT8; break;
        case 16: packetSize = RTC_INTERSECT16; break;
        default: packetSize = RTC_INTERSECT1; break;
    }

    scene = rtcNewScene(RTC_SCENE_STATIC, packetSize);

    int numVerts = this->mesh->vertices.size();
    int numTris = this->mesh->faces.size();

    geomId = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, numTris, numVerts);

    Vector4f *vertices = (Vector4f*) rtcMapBuffer(scene, geomId, RTC_VERTEX_BUFFER);
    for(int i=0; i<numVerts; i++) {
        vertices[i] = mesh->vertices[i];
    }
    rtcUnmapBuffer(scene, geomId, RTC_VERTEX_BUFFER);

    Vector3i *triangles = (Vector3i*) rtcMapBuffer(scene, geomId, RTC_INDEX_BUFFER);
    for(int i=0; i<numTris; i++) {
        gvt::render::data::primitives::Mesh::Face f = mesh->faces[i];
        triangles[i][0] = boost::get<0>(f);
        triangles[i][1] = boost::get<1>(f);
        triangles[i][2] = boost::get<2>(f);
    }
    rtcUnmapBuffer(scene, geomId, RTC_INDEX_BUFFER);

    rtcCommit(scene);

    std::cout << "domain conversion finished" << std::endl;

#if 0
    // Transform mesh
    meshManta = transform<Mesh*, Manta::Mesh*>(this->mesh);
    Manta::Material *material = new Manta::Lambertian(Manta::Color(Manta::RGBColor(0.f, 0.f, 0.f)));
    Manta::MeshTriangle::TriangleType triangleType = Manta::MeshTriangle::KENSLER_SHIRLEY_TRI;

    // Create BVH
    as = new Manta::DynBVH();
    as->setGroup(meshManta);


    //Create Manta
    static Manta::MantaInterface* rtrt = Manta::createManta();

    //Create light set
    Manta::LightSet* lights = new Manta::LightSet();
    lights->add(new Manta::PointLight(Manta::Vector(0, -5, 8), Manta::Color(Manta::RGBColor(1, 1, 1))));

    // Create ambient light
    Manta::AmbientLight* ambient;
    ambient = new Manta::AmbientOcclusionBackground(Manta::Color::white()*0.5, 1, 36);

    //Create context
    Manta::PreprocessContext context(rtrt, 0, 1, lights);
    std::cout << "context.global_lights : " << context.globalLights << std::endl;
    material->preprocess(context);
    as->preprocess(context);


    //Select algorithm
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

#endif
}

EmbreeDomain::EmbreeDomain(std::string filename, gvt::core::math::AffineTransformMatrix<float> m)
: gvt::render::data::domain::GeometryDomain(filename, m)
{
}

EmbreeDomain::EmbreeDomain(const EmbreeDomain& other)
: gvt::render::data::domain::GeometryDomain(other)
{
}

EmbreeDomain::~EmbreeDomain() {
    rtcDeleteGeometry(scene, geomId);
    rtcDeleteScene(scene);
}

bool EmbreeDomain::load() {
    // this function is commented out in MantaDomain
    return true;
}

void EmbreeDomain::free()
{
}

struct parallelTraceE
{
    gvt::render::adapter::embree::data::domain::EmbreeDomain* dom;
    gvt::render::actor::RayVector& rayList;
    gvt::render::actor::RayVector& moved_rays;
    const size_t workSize;

    boost::atomic<size_t>& counter;

    parallelTraceE(
            gvt::render::adapter::embree::data::domain::EmbreeDomain* dom,
            gvt::render::actor::RayVector& rayList,
            gvt::render::actor::RayVector& moved_rays,
            const size_t workSize,
            boost::atomic<size_t>& counter) :
    dom(dom), rayList(rayList), moved_rays(moved_rays), workSize(workSize), counter(counter)
    {
        std::cout << __func__ << ":" << __LINE__ << ": created with workSize: " << workSize << ", raylist size: " << rayList.size() << ", moved rays: " << moved_rays.size() << std::endl;
    }

    void operator()()
    {
        std::cout << "thread starting" << std::endl;
        const size_t maxPacketSize = 64;

        RTCScene scene = dom->getScene();

        gvt::render::actor::RayVector rayPacket;
        gvt::render::actor::RayVector localQueue;
        gvt::render::actor::RayVector localDispatch;

        localQueue.reserve(workSize * 2);
        localDispatch.reserve(rayList.size() * 2);


//                GVT_DEBUG(DBG_ALWAYS, dom->meshManta->vertices.size());
//                GVT_DEBUG(DBG_ALWAYS, dom->meshManta->vertex_indices.size());
//
//                BOOST_FOREACH(int i, dom->meshManta->vertex_indices) {
//                    GVT_DEBUG(DBG_ALWAYS, i);
//                }

        std::cout << "total rays: " << rayList.size() << std::endl;

        while (!rayList.empty())
        {
            boost::unique_lock<boost::mutex> queue(dom->_inqueue);
            std::size_t range = std::min(workSize, rayList.size());
            localQueue.assign(rayList.begin(), rayList.begin() + range);
            rayList.erase(rayList.begin(), rayList.begin() + range);
            queue.unlock();


            GVT_DEBUG(DBG_ALWAYS, "Got " << localQueue.size() << " rays");
            while (!localQueue.empty())
            {
                rayPacket.clear();

                while (rayPacket.size() < GVT_PACKET_SIZE && !localQueue.empty())
                {
                    rayPacket.push_back(localQueue.back());
                    localQueue.pop_back();
                }


                // TODO: assuming packet size == 4

                RTCRay4 ray4 = {};
                for (size_t i = 0; i < rayPacket.size(); i++)
                {
                    Ray r = dom->toLocal(rayPacket[i]);
                    ray4.orgx[i] = r.origin[0];
                    ray4.orgy[i] = r.origin[1];
                    ray4.orgz[i] = r.origin[2];

                    ray4.dirx[i] = r.direction[0];
                    ray4.diry[i] = r.direction[1];
                    ray4.dirz[i] = r.direction[2];
                }

                RTCORE_ALIGN(16) int valid[4] = { -1, -1, -1, -1 };
                rtcIntersect4(valid, scene, ray4);

                //                        GVT_DEBUG(DBG_ALWAYS,"Process packet");

                for (size_t pindex = 0; pindex < rayPacket.size(); pindex++)
                {
                    if (valid[pindex])
                    {
                        //                                GVT_DEBUG(DBG_ALWAYS,"Ray has hit " << pindex);
                        if (rayPacket[pindex].type == gvt::render::actor::Ray::SHADOW)
                        {
                            //                                    GVT_DEBUG(DBG_ALWAYS,"Process ray in shadow");

                            continue;
                        }


                        float t = ray4.tnear[pindex];
                        rayPacket[pindex].t = t;

                        Vector4f embreeNormal = Vector4f(ray4.Ngx[pindex], ray4.Ngy[pindex], ray4.Ngz[pindex], 0.0);
                        Vector4f normal = dom->toWorld(embreeNormal);

                        if (rayPacket[pindex].type == gvt::render::actor::Ray::SECONDARY)
                        {
                            t = (t > 1) ? 1.f / t : t;
                            rayPacket[pindex].w = rayPacket[pindex].w * t;
                        }

                        std::vector<gvt::render::data::scene::Light*> lights = dom->getLights();
                        for (size_t lindex = 0; lindex < lights.size(); lindex++)
                        {
                            gvt::render::actor::Ray ray(rayPacket[pindex]);
                            ray.domains.clear();
                            ray.type = gvt::render::actor::Ray::SHADOW;
                            ray.origin = ray.origin + ray.direction * ray.t;
                            ray.setDirection(lights[lindex]->position - ray.origin);
                            gvt::render::data::Color c = dom->getMesh()->shade(ray, normal, lights[lindex]);
                            //ray.color = COLOR_ACCUM(1.f, c[0], c[1], c[2], 1.f);
                            ray.color = GVT_COLOR_ACCUM(1.f, 1.0, c[1], c[2], 1.f);
                            localQueue.push_back(ray);
                        }

                        int ndepth = rayPacket[pindex].depth - 1;

                        float p = 1.f - (float(rand()) / RAND_MAX);

                        if (ndepth > 0 && rayPacket[pindex].w > p)
                        {
                            gvt::render::actor::Ray ray(rayPacket[pindex]);
                            ray.domains.clear();
                            ray.type = gvt::render::actor::Ray::SECONDARY;
                            ray.origin = ray.origin + ray.direction * ray.t;
                            ray.setDirection(dom->getMesh()->getMaterial()->CosWeightedRandomHemisphereDirection2(normal).normalize());
                            ray.w = ray.w * (ray.direction * normal);
                            ray.depth = ndepth;
                            localQueue.push_back(ray);
                        }
                        //counter++;
                        continue;
                    }
                    //counter++;
                    //GVT_DEBUG(DBG_ALWAYS,"Add to local dispatch");
                    localDispatch.push_back(rayPacket[pindex]);
                }
            }
        }


        GVT_DEBUG(DBG_ALWAYS, "Local dispatch : " << localDispatch.size());

        boost::unique_lock<boost::mutex> moved(dom->_outqueue);
        moved_rays.insert(moved_rays.begin(), localDispatch.begin(), localDispatch.end());
        moved.unlock();
    }
};

void EmbreeDomain::trace(gvt::render::actor::RayVector& rayList, gvt::render::actor::RayVector& moved_rays)
{
    GVT_DEBUG(DBG_ALWAYS, "trace<EmbreeDomain>: " << rayList.size());
    GVT_DEBUG(DBG_ALWAYS, "tracing geometry of domain " << domainID);
    size_t workload = std::max((size_t) 1, (size_t) (rayList.size() / (gvt::core::schedule::asyncExec::instance()->numThreads * 4)));

    std::cout << __PRETTY_FUNCTION__ << ":" << __LINE__ << ": tracing geometry of domain: " << domainID << ", workload: " << workload << std::endl;

    for (size_t rc = 0; rc < gvt::core::schedule::asyncExec::instance()->numThreads; ++rc)
    {
        std::cout << __PRETTY_FUNCTION__ << ":" << __LINE__ << ": creating thread " << rc << std::endl;
        gvt::core::schedule::asyncExec::instance()->run_task(parallelTraceE(this, rayList, moved_rays, workload, counter));
    }

    std::cout << __PRETTY_FUNCTION__ << ":" << __LINE__ << ": " << gvt::core::schedule::asyncExec::instance()->numThreads << " threads created, blocking until finished" << std::endl;
    gvt::core::schedule::asyncExec::instance()->sync();
    //            parallelTrace(this, rayList, moved_rays, rayList.size(),counter)();

#ifdef NDEBUG
    std::cout << "Proccessed rays : " << counter << std::endl;
#else
    GVT_DEBUG(DBG_ALWAYS, "Proccessed rays : " << counter);
#endif
    GVT_DEBUG(DBG_ALWAYS, "Forwarding rays : " << moved_rays.size());
    rayList.clear();
}



