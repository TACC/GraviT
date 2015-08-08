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
#include <gvt/render/adapter/embree/Wrapper.h>
#include <gvt/render/algorithm/Tracers.h>
#include <gvt/render/data/scene/gvtCamera.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/data/Primitives.h>

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
using namespace gvt::render::adapter::embree::data::domain;


int main(int argc, char** argv) {
    // RenderContext::CreateContext() ;
    gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
	if(cntxt == NULL) {
		std::cout << "Something went wrong initializing the context" << std::endl;
		exit(0);
	}

	gvt::core::DBNodeH root = cntxt->getRootNode();

//
//	our friend the cone.
//
	Point4f points[7];
    points[0] = Point4f(0.5,0.0,0.0,1.0);
	points[1] = Point4f(-0.5,0.5,0.0,1.0);
	points[2] = Point4f(-0.5,0.25,0.433013,1.0);
	points[3] = Point4f(-0.5,-0.25,0.43013,1.0);
	points[4] = Point4f(-0.5,-0.5,0.0,1.0);
	points[5] = Point4f(-0.5,-0.25,-0.433013,1.0);
	points[6] = Point4f(-0.5,0.25,-0.433013,1.0);

//
//	build a mesh object from cone geometry.
//
	Mesh* objMesh = new Mesh(new Lambert(Vector4f(0.5,0.5,0.5,1.0)));
	objMesh->addVertex(points[0]);
	objMesh->addVertex(points[1]);
	objMesh->addVertex(points[2]);
	objMesh->addVertex(points[3]);
	objMesh->addVertex(points[4]);
	objMesh->addVertex(points[5]);
	objMesh->addVertex(points[6]);

	objMesh->addFace(1,2,3);
	objMesh->addFace(1,3,4);
	objMesh->addFace(1,4,5);
	objMesh->addFace(1,5,6);
	objMesh->addFace(1,6,7);
	objMesh->addFace(1,7,2);

	objMesh->generateNormals();

    // calculate the bounding box of the mesh
	Point4f lower = points[0], upper = points[0];
	for(int i=1; i<7; i++) {
		for(int j=0; j<3; j++) {
			lower[j] = (lower[j] < points[i][j]) ? lower[j] : points[i][j];
			upper[j] = (upper[j] > points[i][j]) ? upper[j] : points[i][j];
		}
	}
	Box3D *meshbbox = new gvt::render::data::primitives::Box3D(lower, upper);

    // add Mesh to the database
	gvt::core::DBNodeH meshesnode = cntxt->createNodeFromType("Meshes", "Meshes", root.UUID());
	gvt::core::DBNodeH meshnode = cntxt->createNodeFromType("Mesh", "conemesh", meshesnode.UUID());
	meshnode["file"] = string("/fake/path/to/cone");
	meshnode["bbox"] = meshbbox;
	meshnode["ptr"] = objMesh;

    // create instances
	gvt::core::DBNodeH instancesnode = cntxt->createNodeFromType("Instances", "Instances", root.UUID());

    // create a 3x3 grid of cones, offset using i and j
    int instId = 0;
    for(int i=-1; i<2; i++) {
        for(int j=-1; j<2; j++) {
            gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance", "coneinst", instancesnode.UUID());
            instnode["id"] = instId++;
            instnode["meshRef"] = meshnode.UUID();
            auto m = new gvt::core::math::AffineTransformMatrix<float>(true);
            auto minv = new gvt::core::math::AffineTransformMatrix<float>(true);
            auto normi = new gvt::core::math::Matrix3f();

            *m = *m * gvt::core::math::AffineTransformMatrix<float>::createTranslation(0.0, i*0.5, j*0.5);
            *m = *m * gvt::core::math::AffineTransformMatrix<float>::createScale(0.5, 0.5, 0.5);

            instnode["mat"] = m;
            *minv = m->inverse();
            instnode["matInv"] = minv;
            *normi = m->upper33().inverse().transpose();
            instnode["normi"] = normi;

            auto il = (*m) * lower; // TODO: verify that this bbox calc is correct
            auto ih = (*m) * upper;
            Box3D *ibox = new gvt::render::data::primitives::Box3D(il, ih);

            instnode["bbox"] = ibox;
            instnode["centroid"] = ibox->centroid();
        }
    }


	MPI_Init(&argc, &argv);

//
//	Add a point light at 1,1,1 and add it to the database.
//
	gvt::core::DBNodeH lightsnode = cntxt->createNodeFromType("Lights", "Lights", root.UUID());
	gvt::core::DBNodeH lightnode = cntxt->createNodeFromType("PointLight", "conelight", lightsnode.UUID());
	lightnode["position"] = Vector4f(1.0, 1.0, 1.0, 0.0);
	lightnode["color"] = Vector4f(1.0, 1.0, 1.0, 0.0);

//
//	need a camera... using gvtCamera instead of default camera.... because I know how it works.
//
	gvtPerspectiveCamera mycamera;
	Point4f cameraposition(1.0,1.0,1.0,1.0);
	Point4f focus(0.0,0.0,0.0,1.0);
	float fov = 45.0 * M_PI/180.0;
	Vector4f up(0.0,1.0,0.0,0.0);
	mycamera.lookAt(cameraposition,focus,up);
	mycamera.setFOV(fov);
	mycamera.setFilmsize(512,512);

	// Create an object to hold the image.
	Image myimage(mycamera.getFilmSizeWidth(),mycamera.getFilmSizeHeight(),"cone");

	gvt::core::DBNodeH camnode = cntxt->createNodeFromType("Camera","conecam",root.UUID());
	camnode["eyePoint"] = mycamera.getEyePoint();
	camnode["focus"] = mycamera.getFocalPoint();
	camnode["upVector"] = mycamera.getUpVector();

	gvt::core::DBNodeH filmnode = cntxt->createNodeFromType("Film","conefilm",root.UUID());
	filmnode["width"] = mycamera.getFilmSizeWidth();
	filmnode["height"] = mycamera.getFilmSizeHeight();

	gvt::core::DBNodeH schednode = cntxt->createNodeFromType("Schedule","conesched",root.UUID());
	schednode["type"] = gvt::render::scheduler::Image;
	schednode["adapter"] = gvt::render::adapter::Embree;

    std::cout << "\n-- db tree --" << std::endl;
	cntxt->database()->printTree(root.UUID(),10,std::cout);
    std::cout << "\n-- ------- --\n" << std::endl;

//
//	Render it....
//
    // NOTE: later this can be wrapped in a gvt::start() like function

	mycamera.AllocateCameraRays();
	mycamera.generateRays();

	int schedType = gvt::core::variant_toInteger(root["Schedule"]["type"].value());
    switch(schedType) {
        case gvt::render::scheduler::Image :
            {
                std::cout << "starting image scheduler" << std::endl;
                gvt::render::algorithm::Tracer<ImageScheduler>(mycamera.rays,myimage)();
                break;
            }
        case gvt::render::scheduler::Domain :
            {
                std::cout << "skipping domain scheduler" << std::endl;
                //std::cout << "starting domain scheduler" << std::endl;
                //gvt::render::algorithm::Tracer<DomainScheduler>(mycamera.rays,myimage)();
                break;
            }
        default:
            {
                std::cout << "unknown schedule type provided: " << schedType << std::endl;
                break;
            }
    }

	myimage.Write();
}
