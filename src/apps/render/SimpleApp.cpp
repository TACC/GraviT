//
// Simple gravit application.
// Load some geometry and render it.
//
#include <gvt/render/RenderContext.h>
#include <gvt/render/Types.h>
#include <vector>
#include <algorithm>
#include <set>
#include <gvt/core/mpi/Wrapper.h>
#include <gvt/core/Math.h>
#include <gvt/render/data/Dataset.h>
#include <gvt/render/data/Domains.h>
#include <gvt/render/Schedulers.h>
//#include <gvt/render/adapter/manta/Wrapper.h>
//#include <gvt/render/adapter/optix/Wrapper.h>
//#include <gvt/render/adapter/embree/Wrapper.h>
#ifdef GVT_USE_MPE
#include "mpe.h"
#endif
#include <gvt/render/algorithm/Tracers.h>
#include <gvt/render/data/scene/gvtCamera.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/data/Primitives.h>

#include <boost/range/algorithm.hpp>

#include <iostream>

using namespace std;
using namespace gvt::render;
using namespace gvt::core::math;
using namespace gvt::core::mpi;
using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;
using namespace gvt::render::data::primitives;
//using namespace gvt::render::adapter::manta::data::domain;
//using namespace gvt::render::adapter::optix::data::domain;
//using namespace gvt::render::adapter::embree::data::domain;

void test_bvh(gvtPerspectiveCamera &camera);

int main(int argc, char** argv) {


	MPI_Init(&argc, &argv);
	MPI_Pcontrol(0);
	int rank = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef GVT_USE_MPE
	//MPE_Init_log();
	int readstart, readend;
	int renderstart, renderend;
	MPE_Log_get_state_eventIDs(&readstart,&readend);
	MPE_Log_get_state_eventIDs(&renderstart,&renderend);
	if(rank == 0) {
		MPE_Describe_state(readstart, readend, "Initialize context state","red");
		MPE_Describe_state(renderstart, renderend, "Render","yellow");
	}
  MPI_Pcontrol(1);
	MPE_Log_event(readstart,0,NULL);
#endif
    gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
	if(cntxt == NULL) {
		std::cout << "Something went wrong initializing the context" << std::endl;
		exit(0);
	}

	gvt::core::DBNodeH root = cntxt->getRootNode();

    // mix of cones and cubes

    // TODO: maybe rename to 'Data' - as it can store different types of data [mesh, volume, lines]
    gvt::core::DBNodeH dataNodes = cntxt->createNodeFromType("Data", "Data", root.UUID());

    gvt::core::DBNodeH coneMeshNode = cntxt->createNodeFromType("Mesh", "conemesh", dataNodes.UUID());
    {
        Mesh* mesh = new Mesh(new Lambert(Vector4f(0.5,0.5,0.5,1.0)));
        int numPoints = 7;
        Point4f points[7];
        points[0] = Point4f(0.5,0.0,0.0,1.0);
        points[1] = Point4f(-0.5,0.5,0.0,1.0);
        points[2] = Point4f(-0.5,0.25,0.433013,1.0);
        points[3] = Point4f(-0.5,-0.25,0.43013,1.0);
        points[4] = Point4f(-0.5,-0.5,0.0,1.0);
        points[5] = Point4f(-0.5,-0.25,-0.433013,1.0);
        points[6] = Point4f(-0.5,0.25,-0.433013,1.0);

        for(int i=0; i<numPoints; i++) {
            mesh->addVertex(points[i]);
        }
        mesh->addFace(1,2,3);
        mesh->addFace(1,3,4);
        mesh->addFace(1,4,5);
        mesh->addFace(1,5,6);
        mesh->addFace(1,6,7);
        mesh->addFace(1,7,2);
        mesh->generateNormals();

        // calculate bbox
        Point4f lower = points[0], upper = points[0];
        for(int i=1; i<numPoints; i++) {
            for(int j=0; j<3; j++) {
                lower[j] = (lower[j] < points[i][j]) ? lower[j] : points[i][j];
                upper[j] = (upper[j] > points[i][j]) ? upper[j] : points[i][j];
            }
        }
        Box3D *meshbbox = new gvt::render::data::primitives::Box3D(lower, upper);

        // add cone mesh to the database
        coneMeshNode["file"] = string("/fake/path/to/cone");
        coneMeshNode["bbox"] = meshbbox;
        coneMeshNode["ptr"] = mesh;
    }

    gvt::core::DBNodeH cubeMeshNode = cntxt->createNodeFromType("Mesh", "cubemesh", dataNodes.UUID());
    {
        Mesh* mesh = new Mesh(new Lambert(Vector4f(0.5,0.5,0.5,1.0)));
        int numPoints = 8;
        Point4f points[8];
        points[0] = Point4f(-0.5, -0.5,  0.5, 1.0);
        points[1] = Point4f( 0.5, -0.5,  0.5, 1.0);
        points[2] = Point4f( 0.5,  0.5,  0.5, 1.0);
        points[3] = Point4f(-0.5,  0.5,  0.5, 1.0);
        points[4] = Point4f(-0.5, -0.5, -0.5, 1.0);
        points[5] = Point4f( 0.5, -0.5, -0.5, 1.0);
        points[6] = Point4f( 0.5,  0.5, -0.5, 1.0);
        points[7] = Point4f(-0.5,  0.5, -0.5, 1.0);

        for(int i=0; i<numPoints; i++) {
            mesh->addVertex(points[i]);
        }
        // faces are 1 indexed
        mesh->addFace(1, 2, 3);
        mesh->addFace(1, 3, 4);
        mesh->addFace(2, 6, 7);
        mesh->addFace(2, 7, 3);
        mesh->addFace(6, 5, 8);
        mesh->addFace(6, 8, 7);
        mesh->addFace(5, 1, 4);
        mesh->addFace(5, 4, 8);
        mesh->addFace(1, 2, 6);
        mesh->addFace(1, 6, 5);
        mesh->addFace(4, 3, 7);
        mesh->addFace(4, 7, 8);
        mesh->generateNormals();

        // calculate bbox
        Point4f lower = points[0], upper = points[0];
        for(int i=1; i<numPoints; i++) {
            for(int j=0; j<3; j++) {
                lower[j] = (lower[j] < points[i][j]) ? lower[j] : points[i][j];
                upper[j] = (upper[j] > points[i][j]) ? upper[j] : points[i][j];
            }
        }
        Box3D *meshbbox = new gvt::render::data::primitives::Box3D(lower, upper);

        // add cube mesh to the database
        cubeMeshNode["file"] = string("/fake/path/to/cube");
        cubeMeshNode["bbox"] = meshbbox;
        cubeMeshNode["ptr"] = mesh;
    }


	gvt::core::DBNodeH instNodes = cntxt->createNodeFromType("Instances", "Instances", root.UUID());

    // create a NxM grid of alternating cones / cubes, offset using i and j
    int instId = 0;
    int ii[2] = {-2, 3}; // i range
    int jj[2] = {-2, 3}; // j range
    for(int i=ii[0]; i<ii[1]; i++) {
        for(int j=jj[0]; j<jj[1]; j++) {
            gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance", "inst", instNodes.UUID());
            gvt::core::DBNodeH meshNode = (instId % 2) ? coneMeshNode : cubeMeshNode;
            Box3D *mbox = gvt::core::variant_toBox3DPtr(meshNode["bbox"].value());

            instnode["id"] = instId++;
            instnode["meshRef"] = meshNode.UUID();

            auto m = new gvt::core::math::AffineTransformMatrix<float>(true);
            auto minv = new gvt::core::math::AffineTransformMatrix<float>(true);
            auto normi = new gvt::core::math::Matrix3f();
            *m = *m * gvt::core::math::AffineTransformMatrix<float>::createTranslation(0.0, i*0.5, j*0.5);
            *m = *m * gvt::core::math::AffineTransformMatrix<float>::createScale(0.4, 0.4, 0.4);
            instnode["mat"] = m;
            *minv = m->inverse();
            instnode["matInv"] = minv;
            *normi = m->upper33().inverse().transpose();
            instnode["normi"] = normi;

            auto il = (*m) * mbox->bounds[0];
            auto ih = (*m) * mbox->bounds[1];
            Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);
            instnode["bbox"] = ibox;
            instnode["centroid"] = ibox->centroid();
        }
    }



    // add lights, camera, and film to the database
    gvt::core::DBNodeH lightNodes = cntxt->createNodeFromType("Lights", "Lights", root.UUID());
    gvt::core::DBNodeH lightNode = cntxt->createNodeFromType("PointLight", "conelight", lightNodes.UUID());
    lightNode["position"] = Vector4f(1.0, 1.0, 1.0, 0.0);
    lightNode["color"] = Vector4f(0.0, 1.0, 1.0, 0.0);

    // second light just for fun
    gvt::core::DBNodeH lN2 = cntxt->createNodeFromType("PointLight", "conelight", lightNodes.UUID());
    lN2["position"] = Vector4f(2.0, 2.0, 0.0, 0.0);
    lN2["color"] = Vector4f(0.0, 1.0, 0.0, 0.0);

	gvt::core::DBNodeH camNode = cntxt->createNodeFromType("Camera","conecam",root.UUID());
	camNode["eyePoint"] = Point4f(1.0, 1.0, 1.0, 1.0);
	camNode["focus"] = Point4f(0.0, 0.0, 0.0, 1.0);
	camNode["upVector"] = Vector4f(0.0, 1.0, 0.0, 0.0);
	camNode["fov"] = (float)(45.0 * M_PI/180.0);

	gvt::core::DBNodeH filmNode = cntxt->createNodeFromType("Film","conefilm",root.UUID());
	filmNode["width"] = 512;
	filmNode["height"] = 512;

    // TODO: schedule db design could be modified a bit
	gvt::core::DBNodeH schedNode = cntxt->createNodeFromType("Schedule","conesched",root.UUID());
	//schedNode["type"] = gvt::render::scheduler::Image;
	schedNode["type"] = gvt::render::scheduler::Domain;
	schedNode["adapter"] = gvt::render::adapter::Embree;

#if 0
    std::cout << "\n-- db tree --" << std::endl;
	cntxt->database()->printTree(root.UUID(),10,std::cout);
    std::cout << "\n-- ------- --\n" << std::endl;
#endif

    // end db setup


    // use db to create structs needed by system

    // setup gvtCamera from database entries
	gvtPerspectiveCamera mycamera;
	Point4f cameraposition = gvt::core::variant_toPoint4f(camNode["eyePoint"].value());
	Point4f focus = gvt::core::variant_toPoint4f(camNode["focus"].value());
	float fov = gvt::core::variant_toFloat(camNode["fov"].value());
	Vector4f up = gvt::core::variant_toVector4f(camNode["upVector"].value());
	mycamera.lookAt(cameraposition,focus,up);
	mycamera.setFOV(fov);
	mycamera.setFilmsize(gvt::core::variant_toInteger(filmNode["width"].value()), gvt::core::variant_toInteger(filmNode["height"].value()));

#ifdef GVT_USE_MPE
	MPE_Log_event(readend,0,NULL);
#endif
	// setup image from database sizes
	Image myimage(mycamera.getFilmSizeWidth(),mycamera.getFilmSizeHeight(),"cone");

	mycamera.AllocateCameraRays();
	mycamera.generateRays();

	int schedType = gvt::core::variant_toInteger(root["Schedule"]["type"].value());
    switch(schedType) {
        case gvt::render::scheduler::Image :
            {
                std::cout << "starting image scheduler" << std::endl;
                for(int z=0; z<1; z++) {
                    mycamera.AllocateCameraRays();
                    mycamera.generateRays();
                    gvt::render::algorithm::Tracer<ImageScheduler>(mycamera.rays,myimage)();
                }
                break;
            }
        case gvt::render::scheduler::Domain :
            {
                std::cout << "starting domain scheduler" << std::endl;
#ifdef GVT_USE_MPE
								MPE_Log_event(renderstart,0,NULL);
#endif
                gvt::render::algorithm::Tracer<DomainScheduler>(mycamera.rays,myimage)();
#ifdef GVT_USE_MPE
								MPE_Log_event(renderend,0,NULL);
#endif
                break;
            }
        default:
            {
                std::cout << "unknown schedule type provided: " << schedType << std::endl;
                break;
            }
    }

	myimage.Write();
#ifdef GVT_USE_MPE
	MPE_Log_sync_clocks();
	//MPE_Finish_log("gvtSimplelog");
#endif
	if(MPI::COMM_WORLD.Get_size() > 1) MPI_Finalize();
}


// bvh intersection list test
void test_bvh(gvtPerspectiveCamera &mycamera) {
    gvt::core::DBNodeH root = gvt::render::RenderContext::instance()->getRootNode();

    cout << "\n-- bvh test --" << endl;

    auto ilist = root["Instances"].getChildren();
    auto bvh = new gvt::render::data::accel::BVH(ilist);

    // list of rays to test
    std::vector<gvt::render::actor::Ray> rays;
    rays.push_back(mycamera.rays[100*512 + 100]);
    rays.push_back(mycamera.rays[182*512 + 182]);
    rays.push_back(mycamera.rays[256*512 + 256]);
    auto dir = (gvt::core::math::Vector4f(0.0, 0.0, 0.0, 0.0) - gvt::core::math::Vector4f(1.0, 1.0, 1.0, 0.0)).normalize();
    rays.push_back(gvt::render::actor::Ray(gvt::core::math::Point4f(1.0, 1.0, 1.0, 1.0), dir));
    rays.push_back(mycamera.rays[300*512 + 300]);
    rays.push_back(mycamera.rays[400*512 + 400]);
    rays.push_back(mycamera.rays[470*512 + 470]);
    rays.push_back(gvt::render::actor::Ray(gvt::core::math::Point4f(0.0, 0.0, 1.0, 1.0), gvt::core::math::Vector4f(0.0, 0.0, -1.0, 1.0)));
    rays.push_back(mycamera.rays[144231]);

    // test rays and print out which instances were hit
    for(size_t z=0; z<rays.size(); z++) {
        gvt::render::actor::Ray &r = rays[z];
        cout << "bvh: r[" << z << "]: " << r << endl;

        gvt::render::actor::isecDomList& isect = r.domains;
        bvh->intersect(r, isect);
        boost::sort(isect);
        cout << "bvh: r[" << z << "]: isect[" << isect.size() << "]: ";
        for(auto i : isect) { cout << i.domain << " "; }
        cout << endl;
    }

#if 0
    cout << "- check all rays" << endl;
    for(int z=0; z<mycamera.rays.size(); z++) {
        gvt::render::actor::Ray &r = mycamera.rays[z];

        gvt::render::actor::isecDomList& isect = r.domains;
        bvh->intersect(r, isect);
        boost::sort(isect);

        if(isect.size() > 1) {
            cout << "bvh: r[" << z << "]: " << r << endl;
            cout << "bvh: r[" << z << "]: isect[" << isect.size() << "]: ";
            for(auto i : isect) { cout << i.domain << " "; }
            cout << endl;
        }
    }
#endif

    cout << "--------------\n\n" << endl;
}
