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

// TODO: add logic for other packet sizes
#define GVT_PACKET_SIZE 4

using namespace gvt::render::actor;
using namespace gvt::render::adapter::embree::data;
using namespace gvt::render::adapter::embree::data::domain;
using namespace gvt::render::data::domain;
using namespace gvt::render::data::primitives;
using namespace gvt::core::math;

static boost::atomic<size_t> counter(0);

bool EmbreeDomain::init = false;

struct embVertex { float x, y, z, a; };
struct embTriangle { int v0, v1, v2; };

EmbreeDomain::EmbreeDomain(GeometryDomain* domain) : GeometryDomain(*domain)
{
    GVT_DEBUG(DBG_ALWAYS, "Converting domain");

    if(!EmbreeDomain::init) {
        rtcInit(0);
        EmbreeDomain::init = true;
    }

    this->mesh->generateNormals();

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

    embVertex *vertices = (embVertex*) rtcMapBuffer(scene, geomId, RTC_VERTEX_BUFFER);
    for(int i=0; i<numVerts; i++) {
        vertices[i].x = mesh->vertices[i][0];
        vertices[i].y = mesh->vertices[i][1];
        vertices[i].z = mesh->vertices[i][2];
    }
    rtcUnmapBuffer(scene, geomId, RTC_VERTEX_BUFFER);

    embTriangle *triangles = (embTriangle*) rtcMapBuffer(scene, geomId, RTC_INDEX_BUFFER);
    for(int i=0; i<numTris; i++) {
        gvt::render::data::primitives::Mesh::Face f = mesh->faces[i];
        triangles[i].v0 = f.get<0>();
        triangles[i].v1 = f.get<1>();
        triangles[i].v2 = f.get<2>();
    }
    rtcUnmapBuffer(scene, geomId, RTC_INDEX_BUFFER);

    rtcCommit(scene);
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
    }

    void operator()()
    {
        RTCScene scene = dom->getScene();

        gvt::render::actor::RayVector rayPacket;
        gvt::render::actor::RayVector localQueue;
        gvt::render::actor::RayVector localDispatch;

        localQueue.reserve(workSize * 2);
        localDispatch.reserve(rayList.size() * 2);

        while (!rayList.empty())
        {
            boost::unique_lock<boost::mutex> queue(dom->_inqueue);
            std::size_t range = std::min(workSize, rayList.size());
            localQueue.assign(rayList.begin(), rayList.begin() + range);
            rayList.erase(rayList.begin(), rayList.begin() + range);
            queue.unlock();


            // GVT_DEBUG(DBG_ALWAYS, "Got " << localQueue.size() << " rays");
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
                RTCORE_ALIGN(16) int valid[4] = {0};
                for (size_t i = 0; i < rayPacket.size(); i++)
                {
                    Ray r = dom->toLocal(rayPacket[i]);
                    ray4.orgx[i] = r.origin[0];
                    ray4.orgy[i] = r.origin[1];
                    ray4.orgz[i] = r.origin[2];
                    ray4.dirx[i] = r.direction[0];
                    ray4.diry[i] = r.direction[1];
                    ray4.dirz[i] = r.direction[2];
                    ray4.tnear[i] = 0;
                    ray4.tfar[i] = FLT_MAX;
                    ray4.geomID[i] = RTC_INVALID_GEOMETRY_ID;
                    ray4.primID[i] = RTC_INVALID_GEOMETRY_ID;
                    ray4.instID[i] = RTC_INVALID_GEOMETRY_ID;
                    ray4.mask[i] = -1;
                    ray4.time[i] = 0;
                    valid[i] = -1;
                }

                rtcIntersect4(valid, scene, ray4);

                // GVT_DEBUG(DBG_ALWAYS,"Process packet");

                for (size_t pindex = 0; pindex < rayPacket.size(); pindex++)
                {
                    if (valid[pindex] && ray4.geomID[pindex] != (int)RTC_INVALID_GEOMETRY_ID)
                    {
                        // GVT_DEBUG(DBG_ALWAYS,"Ray has hit " << pindex);
                        if (rayPacket[pindex].type == gvt::render::actor::Ray::SHADOW)
                        {
                            // GVT_DEBUG(DBG_ALWAYS,"Process ray in shadow");
                            continue;
                        }

                        float t = ray4.tfar[pindex];
                        rayPacket[pindex].t = t;

                        // for some reason the embree normals aren't working, so just going to manually calculate the triangle normal
                        //Vector4f embreeNormal = Vector4f(ray4.Ngx[pindex], ray4.Ngy[pindex], ray4.Ngz[pindex], 0.0);

                        Vector4f manualNormal;
                        {
                            const int triangle_id = ray4.primID[pindex];
                            const float u = ray4.u[pindex];
                            const float v = ray4.v[pindex];
                            const Mesh::FaceToNormals &normals =
                                dom->getMesh()->faces_to_normals[triangle_id];
                            const Vector4f &a = dom->getMesh()->normals[normals.get<1>()];
                            const Vector4f &b = dom->getMesh()->normals[normals.get<2>()];
                            const Vector4f &c = dom->getMesh()->normals[normals.get<0>()];
                            manualNormal = a * u + b * v + c * (1.0f - u - v);
                            manualNormal = dom->localToWorldNormal(manualNormal);
                            manualNormal.normalize();
                        }

                        const Vector4f &normal = manualNormal;

                        if (rayPacket[pindex].type == gvt::render::actor::Ray::SECONDARY)
                        {
                            t = (t > 1) ? 1.f / t : t;
                            rayPacket[pindex].w = rayPacket[pindex].w * t;
                        }

                        // generate shadow rays
                        std::vector<gvt::render::data::scene::Light*> lights = dom->getLights();
                        for (size_t lindex = 0; lindex < lights.size(); lindex++)
                        {
                            Ray shadow_ray(rayPacket[pindex]);
                            shadow_ray.domains.clear();
                            shadow_ray.type = Ray::SHADOW;
                            // Try to ensure that the shadow ray is on the correct side of the triangle.
                            // Technique adapted from "Robust BVH Ray Traversal" by Thiago Ize.
                            // Using about 8 * ULP(t).
                            float multiplier = 1.0f - 16.0f * std::numeric_limits<float>::epsilon();
                            float t_shadow = multiplier * shadow_ray.t;
                            shadow_ray.origin = shadow_ray.origin + shadow_ray.direction * t_shadow;
                            Vector4f light_position(lights[lindex]->position);
                            Vector4f dir = light_position - shadow_ray.origin;
                            shadow_ray.t_max = dir.length();
                            dir.normalize();
                            shadow_ray.setDirection(dir);
                            gvt::render::data::Color c = dom->getMesh()->shadeFace(ray4.primID[pindex], shadow_ray, normal, lights[lindex]);
                            //gvt::render::data::Color c = dom->getMesh()->mat->shade(shadow_ray, normal, lights[lindex]);
                            shadow_ray.color = GVT_COLOR_ACCUM(1.0f, c[0], c[1], c[2], 1.0f);
                            localQueue.push_back(shadow_ray);
                            //GVT_DEBUG(DBG_ALWAYS, "SHADE_FACE");
                        }

                        int ndepth = rayPacket[pindex].depth - 1;

                        float p = 1.f - (float(rand()) / RAND_MAX);

                        // generate secondary ray
                        if (ndepth > 0 && rayPacket[pindex].w > p)
                        {
                            gvt::render::actor::Ray ray(rayPacket[pindex]);
                            ray.domains.clear();
                            ray.type = gvt::render::actor::Ray::SECONDARY;
                            float multiplier = 1.0f - 16.0f * std::numeric_limits<float>::epsilon();
                            float t_secondary = multiplier * ray.t;
                            ray.origin = ray.origin + ray.direction * t_secondary;
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

    for (size_t rc = 0; rc < gvt::core::schedule::asyncExec::instance()->numThreads; ++rc)
    {
        gvt::core::schedule::asyncExec::instance()->run_task(parallelTraceE(this, rayList, moved_rays, workload, counter));
    }

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



