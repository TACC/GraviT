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
#include <gvt/render/adapter/manta/Wrapper.h>
#include <gvt/render/adapter/optix/Wrapper.h>
#include <gvt/render/algorithm/Tracers.h>
#include <gvt/render/data/scene/gvtCamera.h>
#include <gvt/render/data/scene/Image.h>
#include <gvt/render/data/Primitives.h>
//#include <gvt/render/Attributes.h>

#include <iostream>

using namespace std;
using namespace gvt::render;
using namespace gvt::core::math;
using namespace gvt::core::mpi;
using namespace gvt::render::data::scene;
using namespace gvt::render::schedule;
using namespace gvt::render::data::primitives;
using namespace gvt::render::adapter::manta::data::domain;
using namespace gvt::render::adapter::optix::data::domain;


int main(int argc, char** argv) {

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

	MPI_Init(&argc, &argv);
//
//	scene contains the geometry domain and the other elements like
//	camera, lights, etc.
//
	gvt::render::data::Dataset scene;
//
//	create a geometry domain and place the mesh inside
//
	gvt::render::data::domain::GeometryDomain* domain = new gvt::render::data::domain::GeometryDomain(objMesh);
	scene.domainSet.push_back(domain );
//
//	Add a point light at 1,1,1 and add it to the domain.
//
	Vector4f   pos(1.0,1.0,1.0,0.0);
	Vector4f color(1.0,1.0,1.0,0.0);
	vector<gvt::render::data::scene::Light*> lightset;
	lightset.push_back(new gvt::render::data::scene::PointLight(pos,color));
	domain->setLights(lightset);
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
    RenderContext::CreateContext() ;
    gvt::core::CoreContext *cntxt = gvt::render::RenderContext::instance();
	gvt::core::DBNodeH root = cntxt->getRootNode();
	gvt::core::Variant V;
	gvt::core::DBNodeH camnode,filmnode,datanode;

	camnode = cntxt->createNodeFromType("Camera","raycamera",root.UUID());
	V = mycamera.getEyePoint();
	camnode["eyePoint"] = V;
	V = mycamera.getFocalPoint();
	camnode["focus"] = V;
	V = mycamera.getUpVector();
	camnode["upVector"] = V;
	filmnode = cntxt->createNodeFromType("Film","rayfilm",root.UUID());
	V = mycamera.getFilmSizeWidth();
	filmnode["width"] = V;
	V = mycamera.getFilmSizeHeight();
	filmnode["height"] = V;
    datanode = cntxt->createNodeFromType("Dataset","dataset",root.UUID());
	//V = gvt::render::Attributes::Image;
	V = gvt::render::scheduler::Image;
    datanode["schedule"] = V;
	//root += cntxt->createNode("schedule",V);
	V = new gvt::render::data::Dataset();
	gvt::core::variant_toDatasetPointer(V)->addDomain(new MantaDomain(domain));
    datanode["Dataset_Pointer"] = V;
	//root += cntxt->createNode("dataset",V);
	//V = gvt::render::Attributes::Manta;
	V = gvt::render::adapter::Manta;
	datanode["render_type"] = V;
	//root += cntxt->createNode("render_type",V);
	V = Vector3f(1.,1.,1.);
    datanode["topology"] = V;

	int width = gvt::core::variant_toInteger(root["Film"]["width"].value());
	std::cout << "this should print the tree " << width << std::endl;
	cntxt->database()->printTree(root.UUID(),10,std::cout);
	gvt::render::Attributes& rta =*(gvt::render::Attributes::instance());
	rta.view.width = mycamera.getFilmSizeWidth();
	rta.view.height = mycamera.getFilmSizeHeight();
	rta.schedule = gvt::render::Attributes::Image;
	rta.render_type = gvt::render::Attributes::Manta;
	rta.dataset = new gvt::render::data::Dataset();
	rta.do_lighting = true;
	rta.dataset->addDomain(new MantaDomain(domain));
//
//	Render it....
//	Hardwire the Manta adapter for this application.
//
	mycamera.AllocateCameraRays();
	mycamera.generateRays();
	gvt::render::algorithm::Tracer<DomainScheduler>(mycamera.rays,myimage)();
	myimage.Write();
}
