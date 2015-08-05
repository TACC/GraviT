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

	Point4f lower = points[0], upper = points[0];
	for(int i=1; i<7; i++) {
		for(int j=0; j<3; j++) {
			lower[j] = (lower[j] < points[i][j]) ? lower[j] : points[i][j];
			upper[j] = (upper[j] > points[i][j]) ? upper[j] : points[i][j];
		}
	}
	Box3D *meshbbox = new gvt::render::data::primitives::Box3D(lower, upper);

	gvt::core::DBNodeH meshesnode = cntxt->createNodeFromType("Meshes", "Meshes", root.UUID());
	gvt::core::DBNodeH meshnode = cntxt->createNodeFromType("Mesh", "conemesh", meshesnode.UUID());
	meshnode["file"] = string("/fake/path/to/cone");
	meshnode["bbox"] = meshbbox;
	meshnode["ptr"] = objMesh;

	gvt::core::DBNodeH instancesnode = cntxt->createNodeFromType("Instances", "Instances", root.UUID());
	gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance", "coneinst0", instancesnode.UUID());
	instnode["id"] = 0;
	instnode["meshRef"] = meshnode.UUID();
	instnode["bbox"] = meshbbox;
	instnode["centroid"] = meshbbox->centroid();
	instnode["transMat"] = 0;

	// TODO: alim: where should the mpi_init go?
	MPI_Init(&argc, &argv);

//
//	scene contains the geometry domain and the other elements like
//	camera, lights, etc.
//
	// gvt::render::data::Dataset scene; // TODO: alim: removing dataset concept

//
//	create a geometry domain and place the mesh inside
//
	// gvt::render::data::domain::GeometryDomain* domain = new gvt::render::data::domain::GeometryDomain(objMesh);
	// scene.domainSet.push_back(domain );

//
//	Add a point light at 1,1,1 and add it to the database.
//
	gvt::core::DBNodeH lightnode;
	lightnode = cntxt->createNodeFromType("PointLight", "conelight", root.UUID());
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

//
//	Create an object to hold the image and a pointer to the raw image data.
//
	Image myimage(mycamera.getFilmSizeWidth(),mycamera.getFilmSizeHeight(),"cone");
	unsigned char *imagebuffer = myimage.GetBuffer();

//
//	Attributes class contains information used by tracer to generate image.
//	This will be replaced by the Context in the future.
//

	gvt::core::Variant V;
	gvt::core::DBNodeH camnode,filmnode,datanode;

// camera
	camnode = cntxt->createNodeFromType("Camera","conecam",root.UUID());
	camnode["eyePoint"] = mycamera.getEyePoint();
	camnode["focus"] = mycamera.getFocalPoint();
	camnode["upVector"] = mycamera.getUpVector();

// film
	filmnode = cntxt->createNodeFromType("Film","conefilm",root.UUID());
	filmnode["width"] = mycamera.getFilmSizeWidth();
	filmnode["height"] = mycamera.getFilmSizeHeight();

// scheduler
	gvt::core::DBNodeH schednode = cntxt->createNodeFromType("Schedule","conesched",root.UUID());
	schednode["type"] = gvt::render::scheduler::Image;
	schednode["adapter"] = gvt::render::adapter::Embree;

#if 0
// dataset
    datanode = cntxt->createNodeFromType("Dataset","coneset",root.UUID());
	V = gvt::render::scheduler::Image;
    datanode["schedule"] = V;
	V = new gvt::render::data::Dataset();
    datanode["Dataset_Pointer"] = V;
	V = gvt::render::adapter::Embree;
	datanode["render_type"] = V;

		std::cout << "this should print the tree " << std::endl;
	cntxt->database()->printTree(root.UUID(),10,std::cout);

    if(gvt::core::variant_toInteger(V) == gvt::render::adapter::Embree) {
		gvt::core::variant_toDatasetPointer(root["Dataset"]["Dataset_Pointer"].value())->addDomain(new EmbreeDomain(domain));
	}
	V = Vector3f(1.,1.,1.);
    datanode["topology"] = V;
	V = gvt::render::accelerator::BVH;
	datanode["accel_type"] = V;
	V = domain->getMesh();
	datanode["Mesh_Pointer"] = V;
#endif

    std::cout << "\n-- db tree --" << std::endl;
	cntxt->database()->printTree(root.UUID(),10,std::cout);
    std::cout << "\n-- ------- --\n" << std::endl;
//
//	Render it....
//
	mycamera.AllocateCameraRays();
	mycamera.generateRays();
	int stype = gvt::core::variant_toInteger(root["Schedule"]["type"].value());
	if(stype == gvt::render::scheduler::Image) {
		std::cout << "starting image scheduler" << std::endl;
		gvt::render::algorithm::Tracer<ImageScheduler>(mycamera.rays,myimage)();
	} else if(stype == gvt::render::scheduler::Domain) {
		std::cout << "skipping domain scheduler" << std::endl;
		//std::cout << "starting domain scheduler" << std::endl;
		//gvt::render::algorithm::Tracer<DomainScheduler>(mycamera.rays,myimage)();
	}
	myimage.Write();
}
