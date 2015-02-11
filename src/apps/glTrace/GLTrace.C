/*
 * GLTrace GraviT rendering with render window display. 
 * Modeled after MPITrace example program. Added glut window calls.
 */

#include <mpi.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Frontend/ConfigFile/ConfigFileLoader.h>
#include <GVT/Data/primitives.h>
#include <GVT/Environment/RayTracerAttributes.h>
#include <GVT/Data/scene/Image.h>
#include <GVT/Environment/Camera.h>
#include <Interface/LightSet.h>
#include <GVT/Tracer/tracers.h>
#include <Backend/Manta/gvtmanta.h>

#include <GL/freeglut.h>

using namespace std;

GLubyte *imagebuffer;
static GLint width, height;
static GLint Height;


// Opengl functions

void dispfunc(void) {
	glClear(GL_COLOR_BUFFER_BIT);
	glRasterPos2i(0,0);
        glDrawPixels(width,height,GL_RGB,GL_UNSIGNED_BYTE,imagebuffer);
        glFlush();
}

void reshape(int w, int h) {
        glViewport(0,0,(GLsizei) w, (GLsizei) h);
        Height = (GLint) h;
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0.0, (GLdouble) w, 0.0, (GLdouble) h);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
}


int main(int argc, char* argv[]) {

// mpi initialization

	int rank = -1;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Barrier(MPI_COMM_WORLD);

// input 

	string filename;

	if(argc > 1) {
		filename = argv[1];
	} else {
		cerr << " application requires input config file" << endl;
		if (MPI::COMM_WORLD.Get_size() > 1)MPI_Finalize();
		exit(1);
	}

	GVT::Frontend::ConfigFileLoader cl(filename);
//
// 	Do Ray Tracing without the explict use of the RayTracer class.
// 	Because RayTracer.RenderImage() writes a file and I dont want to
// 	change that class. 
//
	GVT::Dataset::GVTDataset *scene(&cl.scene);
	GVT::Data::RayVector rays;
	scene->camera.SetCamera(rays,1.0);
	GVT::Env::RayTracerAttributes& rta = *(GVT::Env::RayTracerAttributes::instance());
	rta.dataset = new GVT::Dataset::GVTDataset();
	BOOST_FOREACH(GVT::Domain::Domain* dom, scene->domainSet) {
        	GVT::Domain::GeometryDomain* d = (GVT::Domain::GeometryDomain*)dom;
        	d->lights = scene->lightSet;
        	rta.dataset->addDomain(new GVT::Domain::MantaDomain((GVT::Domain::GeometryDomain*)dom));
    	}	


    	rta.view.width = scene->camera.getFilmSizeWidth();
    	rta.view.height = scene->camera.getFilmSizeHeight();
    	rta.view.camera = scene->camera.getEye();
    	rta.view.focus = scene->camera.getLook();
    	rta.view.up = scene->camera.up;

    	rta.sample_rate = 1.0f;
    	rta.sample_ratio = 1.0f;

    	rta.do_lighting = true;
    	rta.schedule = GVT::Env::RayTracerAttributes::Image;
    	rta.render_type = GVT::Env::RayTracerAttributes::Manta;

    	rta.datafile = "";
	MPI_Barrier(MPI_COMM_WORLD);
	Image image(scene->camera.getFilmSizeWidth(),scene->camera.getFilmSizeHeight(),"spoot");
	rays = scene->camera.MakeCameraRays();
	GVT::Trace::Tracer<GVT::Domain::MantaDomain, MPICOMM, ImageSchedule>(rays, image)();
	
//
// Opengl display stuff goes here
//

	if(rank == 0) { // rank 1 process draws stuff
		width = rta.view.width;
		height = rta.view.height;
		imagebuffer = image.GetBuffer();
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
		glutInitWindowSize(width,height);
		glutInitWindowPosition(10,10);
		glutCreateWindow(filename.c_str());
		glClearColor(0.,0.,0.,0.);
		glShadeModel(GL_FLAT);
		glPixelStorei(GL_UNPACK_ALIGNMENT,1);
		glutDisplayFunc(dispfunc);
		glutReshapeFunc(reshape);
		glutMainLoop();
	}		

	if(MPI::COMM_WORLD.Get_size() > 1 ) MPI_Finalize();
	return 0;

}
