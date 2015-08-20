//
// EmbreeMeshAdapter.cpp
//

#include "gvt/render/adapter/embree/data/EmbreeMeshAdapter.h"

#include "gvt/core/CoreContext.h"

#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>

#include <gvt/core/schedule/TaskScheduling.h> // used for threads

#include <gvt/render/actor/Ray.h>
#include <gvt/render/adapter/embree/data/Transforms.h>

#include <gvt/render/data/scene/ColorAccumulator.h>
#include <gvt/render/data/scene/Light.h>

#include <thread>

#include <boost/atomic.hpp>
#include <boost/foreach.hpp>
#include <boost/timer/timer.hpp>

// TODO: add logic for other packet sizes
#define GVT_EMBREE_PACKET_SIZE 4

using namespace gvt::render::actor;
using namespace gvt::render::adapter::embree::data;
using namespace gvt::render::data::primitives;
using namespace gvt::core::math;

static std::atomic<size_t> counter(0);

bool EmbreeMeshAdapter::init = false;

struct embVertex { float x, y, z, a; };
struct embTriangle { int v0, v1, v2; };

EmbreeMeshAdapter::EmbreeMeshAdapter(gvt::core::DBNodeH node) : Adapter(node)
{
    GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: converting mesh node " << gvt::core::uuid_toString(node.UUID()));

    if(!EmbreeMeshAdapter::init) {
        rtcInit(0);
        EmbreeMeshAdapter::init = true;
    }

    Mesh* mesh = gvt::core::variant_toMeshPtr(node["ptr"].value());

    GVT_ASSERT(mesh, "EmbreeMeshAdapter: mesh pointer in the database is null");

    mesh->generateNormals();

    switch(GVT_EMBREE_PACKET_SIZE) {
        case 4: packetSize = RTC_INTERSECT4; break;
        case 8: packetSize = RTC_INTERSECT8; break;
        case 16: packetSize = RTC_INTERSECT16; break;
        default: packetSize = RTC_INTERSECT1; break;
    }

    scene = rtcNewScene(RTC_SCENE_STATIC, packetSize);

    int numVerts = mesh->vertices.size();
    int numTris = mesh->faces.size();

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

    // TODO: note: embree doesn't save normals in its mesh structure, have to calculate the normal based on uv value
    // later we might have to copy the mesh normals to a local structure so we can correctly calculate the bounced rays

    rtcCommit(scene);
}

EmbreeMeshAdapter::~EmbreeMeshAdapter() {
    rtcDeleteGeometry(scene, geomId);
    rtcDeleteScene(scene);
}

struct parallelTraceE2
{
    /**
     * Pointer to EmbreeMeshAdapter to get Embree scene information
     */
    gvt::render::adapter::embree::data::EmbreeMeshAdapter* adapter;

    /**
     * Shared ray list used in the current trace() call
     */
    gvt::render::actor::RayVector& rayList;

    /**
     * Shared outgoing ray list used in the current trace() call
     */
    gvt::render::actor::RayVector& moved_rays;

    /**
     * Number of rays to work on at once [load balancing].
     */
    const size_t workSize;

    /**
     * Index into the shared `rayList`.  Atomically incremented to 'grab'
     * the next set of rays.
     */
    std::atomic<size_t>& sharedIdx;

    /**
     * DB reference to the current instance
     */
    gvt::core::DBNodeH instNode;

    /**
     * Stored bbox in the current instance
     */
    const gvt::render::data::primitives::Box3D *bbox; // transforms

    /**
     * Stored transformation matrix in the current instance
     */
    const gvt::core::math::AffineTransformMatrix<float> *m;

    /**
     * Stored inverse transformation matrix in the current instance
     */
    const gvt::core::math::AffineTransformMatrix<float> *minv;

    /**
     * Stored upper33 inverse matrix in the current instance
     */
    const gvt::core::math::Matrix3f *normi;

    /**
     * Stored transformation matrix in the current instance
     */
    const std::vector<gvt::render::data::scene::Light*>& lights;

    /**
     * Count the number of rays processed by the current trace() call.
     *
     * Used for debugging purposes
     */
    std::atomic<size_t>& counter;

    /**
     * Thread local outgoing ray queue
     */
    gvt::render::actor::RayVector localDispatch;

    /**
     * List of shadow rays to be processed
     */
    gvt::render::actor::RayVector shadowRays;

    /**
     * Size of Embree packet
     */
    const size_t packetSize; // TODO: later make this configurable

    /**
     * Construct a parallelTraceE2 struct with information needed for the thread
     * to do its tracing
     */
    parallelTraceE2(
            gvt::render::adapter::embree::data::EmbreeMeshAdapter* adapter,
            gvt::render::actor::RayVector& rayList,
            gvt::render::actor::RayVector& moved_rays,
            std::atomic<size_t>& sharedIdx,
            const size_t workSize,
            gvt::core::DBNodeH instNode,
            gvt::render::data::primitives::Box3D *bbox, // transforms
            gvt::core::math::AffineTransformMatrix<float> *m,
            gvt::core::math::AffineTransformMatrix<float> *minv,
            gvt::core::math::Matrix3f *normi,
            std::vector<gvt::render::data::scene::Light*>& lights,
            std::atomic<size_t>& counter) :
        adapter(adapter), rayList(rayList), moved_rays(moved_rays), sharedIdx(sharedIdx), workSize(workSize),
        instNode(instNode), bbox(bbox), m(m), minv(minv), normi(normi), lights(lights),
        counter(counter), packetSize(adapter->getPacketSize())
    {
    }

    /**
     * Convert a set of rays from a vector into a RTCRay4 ray packet.
     *
     * \param ray4          reference of RTCRay4 struct to write to
     * \param valid         aligned array of 4 ints to mark valid rays
     * \param resetValid    if true, reset the valid bits, if false, re-use old valid to know which to convert
     * \param packetSize    number of rays to convert
     * \param rays          vector of rays to read from
     * \param startIdx      starting point to read from in `rays`
     */
    void prepRTCRay4(RTCRay4 &ray4, int valid[4], const bool resetValid, const int packetSize, gvt::render::actor::RayVector& rays, const size_t startIdx) {
        // reset valid to match the number of active rays in the packet
        if(resetValid) {
            for(int i=0; i<packetSize; i++) {
                valid[i] = -1;
            }
            for(int i=packetSize; i<4; i++) {
                valid[i] = 0;
            }
        }

        // convert packetSize rays into embree's RTCRay4 struct
        for(int i=0; i<packetSize; i++) {
            if(valid[i]) {
                const Ray &r = rays[startIdx + i];
                const auto origin = (*minv) * r.origin; // transform ray to local space
                const auto direction = (*minv) * r.direction;
                ray4.orgx[i] = origin[0];
                ray4.orgy[i] = origin[1];
                ray4.orgz[i] = origin[2];
                ray4.dirx[i] = direction[0];
                ray4.diry[i] = direction[1];
                ray4.dirz[i] = direction[2];
                ray4.tnear[i] = 0.0;
                ray4.tfar[i] = FLT_MAX;
                ray4.geomID[i] = RTC_INVALID_GEOMETRY_ID;
                ray4.primID[i] = RTC_INVALID_GEOMETRY_ID;
                ray4.instID[i] = RTC_INVALID_GEOMETRY_ID;
                ray4.mask[i] = -1;
                ray4.time[i] = 0;
            }
        }
    }

    /**
     * Generate shadow rays for a given ray
     *
     * \param r ray to generate shadow rays for
     * \param normal calculated normal
     * \param primId primitive id for shading
     * \param mesh pointer to mesh struct [TEMPORARY]
     */
    void generateShadowRays(const gvt::render::actor::Ray &r, const gvt::core::math::Vector4f &normal, int primID, gvt::render::data::primitives::Mesh* mesh) {
        for(gvt::render::data::scene::Light* light : lights) {
            GVT_ASSERT(light, "generateShadowRays: light is null for some reason");
            // Try to ensure that the shadow ray is on the correct side of the triangle.
            // Technique adapted from "Robust BVH Ray Traversal" by Thiago Ize.
            // Using about 8 * ULP(t).
            const float multiplier = 1.0f - 16.0f * std::numeric_limits<float>::epsilon();
            const float t_shadow = multiplier * r.t;

            const Point4f origin = r.origin + r.direction * t_shadow;
            const Vector4f dir = light->position - origin;
            const float t_max = dir.length();

            // note: ray copy constructor is too heavy, so going to build it manually
            Ray shadow_ray(r.origin + r.direction * t_shadow, dir, r.w, Ray::SHADOW, r.depth);
            shadow_ray.t = r.t;
            shadow_ray.id = r.id;
            shadow_ray.t_max = t_max;

            // FIXME: remove dependency on mesh->shadeFace
            gvt::render::data::Color c = mesh->shadeFace(primID, shadow_ray, normal, light);
            //gvt::render::data::Color c = adapter->getMesh()->mat->shade(shadow_ray, normal, lights[lindex]);
            shadow_ray.color = GVT_COLOR_ACCUM(1.0f, c[0], c[1], c[2], 1.0f);

            // note: still have to pay the cost of the copy constructor here
            shadowRays.push_back(shadow_ray);
        }
    }

    /**
     * Test occlusion for stored shadow rays.  Add missed rays
     * to the dispatch queue.
     */
    void traceShadowRays() {
        RTCScene scene = adapter->getScene();
        RTCRay4 ray4 = {};
        RTCORE_ALIGN(16) int valid[4] = {0};

        for(size_t idx = 0; idx < shadowRays.size(); idx += packetSize) {
            const size_t localPacketSize = (idx + packetSize > shadowRays.size()) ? (shadowRays.size() - idx) : packetSize;

            // create a shadow packet and trace with rtcOccluded
            prepRTCRay4(ray4, valid, true, localPacketSize, shadowRays, idx);
            rtcOccluded4(valid, scene, ray4);

            for(size_t pi = 0; pi < localPacketSize; pi++) {
                auto &r = shadowRays[idx + pi];
                if(valid[pi] && ray4.geomID[pi] == (int)RTC_INVALID_GEOMETRY_ID) {
                    // ray is valid, but did not hit anything, so add to dispatch queue
                    localDispatch.push_back(r);
                }
            }
        }
        shadowRays.clear();
    }

    /**
     * Trace function.
     *
     * Loops through rays in `rayList`, converts them to embree format, and traces against embree's scene
     *
     * TODO: write detailed description
     *
     */
    void operator()()
    {
#ifdef GVT_USE_DEBUG
        boost::timer::auto_cpu_timer t_functor("EmbreeMeshAdapter: thread trace time: %w\n");
#endif
        GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: started thread");

        GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: getting mesh [hack for now]");
        // TODO: don't use gvt mesh. need to figure out way to do per-vertex-normals and shading calculations
        auto mesh = gvt::core::variant_toMeshPtr(instNode["meshRef"].deRef()["ptr"].value());

        RTCScene scene = adapter->getScene();
        localDispatch.reserve(rayList.size() * 2);

        // there is an upper bound on the nubmer of shadow rays generated per embree packet
        // its embree_packetSize * lights.size()
        shadowRays.reserve(packetSize * lights.size());

        GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: starting while loop");

        while (sharedIdx < rayList.size())
        {
#ifdef GVT_USE_DEBUG
            // boost::timer::auto_cpu_timer t_outer_loop("EmbreeMeshAdapter: workSize rays traced: %w\n");
#endif

            // atomically get the next chunk range
            size_t workStart = sharedIdx.fetch_add(workSize);

            // have to double check that we got the last valid chunk range
            if(workStart > rayList.size()) {
                break;
            }

            // calculate the end work range
            size_t workEnd = workStart + workSize;
            if(workEnd > rayList.size()) {
                workEnd = rayList.size();
            }

            RTCRay4 ray4 = {};
            RTCORE_ALIGN(16) int valid[4] = {0};

            GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: working on rays [" << workStart << ", " << workEnd << "]");
            for(size_t localIdx = workStart; localIdx < workEnd; localIdx += packetSize)
            {
                // this is the local packet size. this might be less than the main packetSize due to uneven amount of rays
                const size_t localPacketSize = (localIdx + packetSize > workEnd) ? (workEnd - localIdx) : packetSize;

                // trace a packet of rays, then keep tracing the generated secondary rays to completion
                // tracks to see if there are any valid rays left in the packet, if so, keep tracing
                // NOTE: perf issue: this will cause smaller and smaller packets to be traced at a time - need to track to see effects
                bool validRayLeft = true;

                // the first time we enter the loop, we want to reset the valid boolean list
                // TODO: alternatively just memset the valid to local packet size instead of testing the 'if' every time
                bool resetValid = true;
                while(validRayLeft) {
                    validRayLeft = false;

                    prepRTCRay4(ray4, valid, resetValid, localPacketSize, rayList, localIdx);
                    rtcIntersect4(valid, scene, ray4);
                    resetValid = false;

                    for(size_t pi = 0; pi < localPacketSize; pi++) {
                        if(valid[pi]) {
                            //counter++; // counter tracks rays processed [atomic]
                            auto &r = rayList[localIdx + pi];
                            if(ray4.geomID[pi] != (int)RTC_INVALID_GEOMETRY_ID) {
                                // ray has hit something

                                // shadow ray hit something, so it should be dropped
                                if (r.type == gvt::render::actor::Ray::SHADOW)
                                {
                                    continue;
                                }

                                float t = ray4.tfar[pi];
                                r.t = t;

                                // FIXME: embree does not take vertex normal information, the examples have the application calculate the normal using
                                // math similar to the bottom.  this means we have to keep around a 'faces_to_normals' list along with a 'normals' list
                                // for the embree adapter
                                //
                                // old fixme: fix embree normal calculation to remove dependency from gvt mesh
                                // for some reason the embree normals aren't working, so just going to manually calculate the triangle normal
                                // Vector4f embreeNormal = Vector4f(ray4.Ngx[pi], ray4.Ngy[pi], ray4.Ngz[pi], 0.0);

                                Vector4f manualNormal;
                                {
                                    const int triangle_id = ray4.primID[pi];
                                    const float u = ray4.u[pi];
                                    const float v = ray4.v[pi];
                                    const Mesh::FaceToNormals &normals = mesh->faces_to_normals[triangle_id]; // FIXME: need to figure out where to store `faces_to_normals` list
                                    const Vector4f &a = mesh->normals[normals.get<1>()];
                                    const Vector4f &b = mesh->normals[normals.get<2>()];
                                    const Vector4f &c = mesh->normals[normals.get<0>()];
                                    manualNormal = a * u + b * v + c * (1.0f - u - v);

                                    manualNormal = (*normi) * (gvt::core::math::Vector3f)manualNormal;
                                    manualNormal.normalize();
                                }
                                const Vector4f &normal = manualNormal;

                                // reduce contribution of the color that the shadow rays get
                                if (r.type == gvt::render::actor::Ray::SECONDARY)
                                {
                                    t = (t > 1) ? 1.f / t : t;
                                    r.w = r.w * t;
                                }

                                generateShadowRays(r, normal, ray4.primID[pi], mesh);

                                int ndepth = r.depth - 1;
                                float p = 1.f - (float(rand()) / RAND_MAX);
                                // replace current ray with generated secondary ray
                                if (ndepth > 0 && r.w > p)
                                {
                                    r.domains.clear();
                                    r.type = gvt::render::actor::Ray::SECONDARY;
                                    const float multiplier = 1.0f - 16.0f * std::numeric_limits<float>::epsilon(); // TODO: move out somewhere / make static
                                    const float t_secondary = multiplier * r.t;
                                    r.origin = r.origin + r.direction * t_secondary;

                                    // TODO: remove this dependency on mesh, store material object in the database
                                    //r.setDirection(adapter->getMesh()->getMaterial()->CosWeightedRandomHemisphereDirection2(normal).normalize());
                                    r.setDirection(mesh->getMaterial()->CosWeightedRandomHemisphereDirection2(normal).normalize());

                                    r.w = r.w * (r.direction * normal);
                                    r.depth = ndepth;
                                    validRayLeft = true; // we still have a valid ray in the packet to trace
                                } else {
                                    // secondary ray is terminated, so disable its valid bit
                                    valid[pi] = 0;
                                }
                            } else {
                                // ray is valid, but did not hit anything, so add to dispatch queue and disable it
                                localDispatch.push_back(r);
                                valid[pi] = 0;
                            }
                        }
                    }

                    // trace shadow rays generated by the packet
                    traceShadowRays();
                }
            }
        }

#ifdef GVT_USE_DEBUG
        size_t shadow_count = 0;
        size_t primary_count = 0;
        size_t secondary_count = 0;
        size_t other_count = 0;
        for(auto &r : localDispatch) {
            switch(r.type) {
                case gvt::render::actor::Ray::SHADOW: shadow_count++; break;
                case gvt::render::actor::Ray::PRIMARY: primary_count++; break;
                case gvt::render::actor::Ray::SECONDARY: secondary_count++; break;
                default: other_count++; break;
            }
        }
        GVT_DEBUG(DBG_ALWAYS, "Local dispatch : " << localDispatch.size()
                << ", types: primary: " << primary_count << ", shadow: " << shadow_count << ", secondary: " << secondary_count << ", other: " << other_count);
#endif

        // copy localDispatch rays to outgoing rays queue
        boost::unique_lock<boost::mutex> moved(adapter->_outqueue);
        moved_rays.insert(moved_rays.end(), localDispatch.begin(), localDispatch.end());
        moved.unlock();
    }
};

struct parallelTraceE1
{
    gvt::render::adapter::embree::data::EmbreeMeshAdapter* adapter;
    gvt::core::DBNodeH instNode;
    gvt::render::actor::RayVector& rayList;
    gvt::render::actor::RayVector& moved_rays;
    const size_t workSize;

    std::atomic<size_t>& counter;

    parallelTraceE1(
            gvt::render::adapter::embree::data::EmbreeMeshAdapter* adapter,
            gvt::core::DBNodeH instNode,
            gvt::render::actor::RayVector& rayList,
            gvt::render::actor::RayVector& moved_rays,
            const size_t workSize,
            std::atomic<size_t>& counter) :
    adapter(adapter), instNode(instNode), rayList(rayList), moved_rays(moved_rays), workSize(workSize), counter(counter)
    {
    }

    void operator()()
    {
        GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: starting parallel trace with " << rayList.size() << " rays");
        GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: instance: " << gvt::core::uuid_toString(instNode.UUID()));

        gvt::core::DBNodeH root = gvt::core::CoreContext::instance()->getRootNode();

        // pull out instance data
        GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: getting instance transform data");
        gvt::render::data::primitives::Box3D *bbox = gvt::core::variant_toBox3DPtr(instNode["bbox"].value());
        gvt::core::math::AffineTransformMatrix<float> *m = gvt::core::variant_toAffineTransformMatPtr(instNode["mat"].value());
        gvt::core::math::AffineTransformMatrix<float> *minv = gvt::core::variant_toAffineTransformMatPtr(instNode["matInv"].value());
        gvt::core::math::Matrix3f *normi = gvt::core::variant_toMatrix3fPtr(instNode["normi"].value());

        GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: getting mesh [hack for now]");
        // TODO: don't use gvt mesh, need to fix embree triangle normal hit
        auto mesh = gvt::core::variant_toMeshPtr(instNode["meshRef"].deRef()["ptr"].value());

        // pull out lights list
        auto lightNodes = root["Lights"].getChildren();
        GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: getting lights list [hack for now]: size: " << lightNodes.size());

        RTCScene scene = adapter->getScene();

        gvt::render::actor::RayVector rayPacket;
        gvt::render::actor::RayVector localQueue;
        gvt::render::actor::RayVector localDispatch;

        localQueue.reserve(workSize * 2);
        localDispatch.reserve(rayList.size() * 2);

        GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: starting while loop");

        while (!rayList.empty())
        {
            boost::unique_lock<boost::mutex> queue(adapter->_inqueue);
            std::size_t range = std::min(workSize, rayList.size());
            localQueue.assign(rayList.begin(), rayList.begin() + range);
            rayList.erase(rayList.begin(), rayList.begin() + range);
            queue.unlock();


            //GVT_DEBUG(DBG_ALWAYS, "Got " << localQueue.size() << " rays");
            while (!localQueue.empty())
            {
                rayPacket.clear();

                while (rayPacket.size() < GVT_EMBREE_PACKET_SIZE && !localQueue.empty())
                {
                    rayPacket.push_back(localQueue.back());
                    localQueue.pop_back();
                }


                // TODO: assuming packet size == 4

                RTCRay4 ray4 = {};
                RTCORE_ALIGN(16) int valid[4] = {0};
                for (size_t i = 0; i < rayPacket.size(); i++)
                {
                    Ray r = rayPacket[i];

                    r.origin = (*minv) * r.origin; // 'toLocal'
                    r.direction = (*minv) * r.direction;

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

                //GVT_DEBUG(DBG_ALWAYS,"Process packet");

                for (size_t pindex = 0; pindex < rayPacket.size(); pindex++)
                {
                    if (valid[pindex] && ray4.geomID[pindex] != (int)RTC_INVALID_GEOMETRY_ID)
                    {
                        //GVT_DEBUG(DBG_ALWAYS,"Ray has hit " << pindex);
                        if (rayPacket[pindex].type == gvt::render::actor::Ray::SHADOW)
                        {
                            //GVT_DEBUG(DBG_ALWAYS,"Process ray in shadow");
                            continue;
                        }

                        float t = ray4.tfar[pindex];
                        rayPacket[pindex].t = t;

                        // FIXME: fix embree normal calculation to remove dependency from gvt mesh
                        // for some reason the embree normals aren't working, so just going to manually calculate the triangle normal
                        // Vector4f embreeNormal = Vector4f(ray4.Ngx[pindex], ray4.Ngy[pindex], ray4.Ngz[pindex], 0.0);

                        Vector4f manualNormal;
                        {
                            const int triangle_id = ray4.primID[pindex];
                            const float u = ray4.u[pindex];
                            const float v = ray4.v[pindex];
                            const Mesh::FaceToNormals &normals = mesh->faces_to_normals[triangle_id]; // FIXME: this needs to be removed, cannot assume mesh is live in memory
                            const Vector4f &a = mesh->normals[normals.get<1>()];
                            const Vector4f &b = mesh->normals[normals.get<2>()];
                            const Vector4f &c = mesh->normals[normals.get<0>()];
                            manualNormal = a * u + b * v + c * (1.0f - u - v);

                            // 'localToWorldNormal'
                            //manualNormal = adapter->localToWorldNormal(manualNormal);
                            manualNormal = (*normi) * (gvt::core::math::Vector3f)manualNormal;
                            manualNormal.normalize();
                        }

                        const Vector4f &normal = manualNormal;

                        if (rayPacket[pindex].type == gvt::render::actor::Ray::SECONDARY)
                        {
                            t = (t > 1) ? 1.f / t : t;
                            rayPacket[pindex].w = rayPacket[pindex].w * t;
                        }

                        // generate shadow rays
                        //std::vector<gvt::render::data::scene::Light*> lights = adapter->getLights();
                        //for (size_t lindex = 0; lindex < lights.size(); lindex++)
                        for (auto lightNode : lightNodes)
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
                            Vector4f light_position(gvt::core::variant_toVector4f(lightNode["position"].value())); // TODO: cleanup
                            Vector4f dir = light_position - shadow_ray.origin;
                            shadow_ray.t_max = dir.length();
                            dir.normalize();
                            shadow_ray.setDirection(dir);

                            // TODO: hack for lights / need to cleanup
                            auto pos = gvt::core::variant_toVector4f(lightNode["position"].value());
                            auto color = gvt::core::variant_toVector4f(lightNode["color"].value());
                            auto pl = new gvt::render::data::scene::PointLight(pos, color);

                            gvt::render::data::Color c = mesh->shadeFace(ray4.primID[pindex], shadow_ray, normal, pl);

                            //gvt::render::data::Color c = adapter->getMesh()->mat->shade(shadow_ray, normal, lights[lindex]);
                            shadow_ray.color = GVT_COLOR_ACCUM(1.0f, c[0], c[1], c[2], 1.0f);
                            localQueue.push_back(shadow_ray);
                            ////GVT_DEBUG(DBG_ALWAYS, "SHADE_FACE");
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

                            // TODO: hack: remove this dependency on mesh, store material object in the database
                            //ray.setDirection(adapter->getMesh()->getMaterial()->CosWeightedRandomHemisphereDirection2(normal).normalize());
                            ray.setDirection(mesh->getMaterial()->CosWeightedRandomHemisphereDirection2(normal).normalize());

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

        boost::unique_lock<boost::mutex> moved(adapter->_outqueue);
        moved_rays.insert(moved_rays.begin(), localDispatch.begin(), localDispatch.end());
        moved.unlock();
    }
};

void EmbreeMeshAdapter::trace(gvt::render::actor::RayVector& rayList, gvt::render::actor::RayVector& moved_rays, gvt::core::DBNodeH instNode)
{
#ifdef GVT_USE_DEBUG
    boost::timer::auto_cpu_timer t_functor("EmbreeMeshAdapter: trace time: %w\n");
#endif
    std::atomic<size_t> sharedIdx(0); // shared index into rayList
    const size_t numThreads = gvt::core::schedule::asyncExec::instance()->numThreads;
    const size_t workSize = std::max((size_t) 8, (size_t) (rayList.size() / (numThreads * 8))); // size of 'chunk' of rays to work on

    // std::cout << "EmbreeMeshAdapter: trace: instNode: " << gvt::core::uuid_toString(instNode.UUID()) << ", rays: " << rayList.size() << ", workSize: " << workSize << ", threads: " << numThreads << std::endl;
    GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: trace: instNode: " << gvt::core::uuid_toString(instNode.UUID()) << ", rays: " << rayList.size() << ", workSize: " << workSize << ", threads: " << gvt::core::schedule::asyncExec::instance()->numThreads);

    // pull out information out of the database, create local vectors that will be passed into the parallel struct
    gvt::core::DBNodeH root = gvt::core::CoreContext::instance()->getRootNode();

    // pull out instance data
    GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: getting instance transform data");
    gvt::render::data::primitives::Box3D *bbox = gvt::core::variant_toBox3DPtr(instNode["bbox"].value());
    gvt::core::math::AffineTransformMatrix<float> *m = gvt::core::variant_toAffineTransformMatPtr(instNode["mat"].value());
    gvt::core::math::AffineTransformMatrix<float> *minv = gvt::core::variant_toAffineTransformMatPtr(instNode["matInv"].value());
    gvt::core::math::Matrix3f *normi = gvt::core::variant_toMatrix3fPtr(instNode["normi"].value());


    // TODO: wrap this db light array -> class light array in some sort of helper function
    // pull out lights list and convert into gvt::Lights format for now
    auto lightNodes = root["Lights"].getChildren();
    std::vector<gvt::render::data::scene::Light*> lights;
    lights.reserve(2);
    for(auto lightNode : lightNodes) {
        auto color = gvt::core::variant_toVector4f(lightNode["color"].value());

        if(lightNode.name() == std::string("PointLight")) {
            auto pos = gvt::core::variant_toVector4f(lightNode["position"].value());
            lights.push_back(new gvt::render::data::scene::PointLight(pos, color));
        } else if(lightNode.name() == std::string("AmbientLight")) {
            lights.push_back(new gvt::render::data::scene::AmbientLight(color));
        }
    }
    GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: converted " << lightNodes.size() << " light nodes into structs: size: " << lights.size());


//#define EMBREE_OLD
#define EMBREE_BOOST_THREADS

    // # notes
    // boost threads vs c++11 threads don't seem to have much of a runtime difference

#if 1
  #ifdef EMBREE_BOOST_THREADS
    for (size_t rc = 0; rc < numThreads; ++rc)
    {
    #ifdef EMBREE_OLD
        gvt::core::schedule::asyncExec::instance()->run_task(
                parallelTraceE1(this, instNode, rayList, moved_rays, workSize, counter)
                );
    #else
        gvt::core::schedule::asyncExec::instance()->run_task(
                parallelTraceE2(this, rayList, moved_rays, sharedIdx, workSize, instNode, bbox, m, minv, normi, lights, counter)
                );
    #endif
    }

    gvt::core::schedule::asyncExec::instance()->sync();
  #else
    // c++11 threads
    std::vector<std::thread> threads;
    for(size_t rc = 0; rc < numThreads; rc++) {
    #ifdef EMBREE_OLD
        threads.push_back(std::thread(
                    parallelTraceE1(this, instNode, rayList, moved_rays, workSize, counter)
                    ));

    #else
        threads.push_back(std::thread(
                    parallelTraceE2(this, rayList, moved_rays, sharedIdx, workSize, instNode, bbox, m, minv, normi, lights, counter)
                    ));
    #endif
    }

    for(auto &th : threads) th.join();
  #endif

#else
    // NOTE: this is the sequential call
  #ifdef EMBREE_OLD
    //parallelTraceEOrig(this, instNode, rayList, moved_rays, workSize, counter, lights)();
    parallelTraceE1(this, instNode, rayList, moved_rays, workSize, counter)();
  #else
    parallelTraceE2(this, rayList, moved_rays, sharedIdx, workSize, instNode, bbox, m, minv, normi, lights, counter)();
  #endif
#endif

#ifdef NDEBUG
    //std::cout << "EmbreeMeshAdapter: Processed rays: " << counter << std::endl;
#else
    //GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: Processed rays: " << counter);
#endif

    GVT_DEBUG(DBG_ALWAYS, "EmbreeMeshAdapter: Forwarding rays: " << moved_rays.size());

    rayList.clear();
}



