/**
 * GraviT delaunay tessellation test application two. 
 * This application reads sets of points in vtk unstructured grid format
 * Each file contains a set of points to be tesslated. The application
 * loads the points from a file into a separate GraviT mesh object. Each
 * mesh object is tesslated and rendered by GraviT.
 *
 */
// GraviT includes
#include <gvt/render/api/api.h>
#include <gvt/core/Math.h>
#include <gvt/render/cntx/rcontext.h>
#include <gvt/render/Renderer.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/Types.h>
#include <gvt/render/data/Domains.h>
#ifdef GVT_RENDER_ADAPTER_EMBREE
#include <gvt/render/adapter/embree/EmbreeMeshAdapter.h>
#endif
#include <gvt/render/data/primitives/Material.h>
// this application calls the qhull tessellation library directly to create
// a set of 3D points. Include the necessary includes for that here.
#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/RboxPoints.h"
#include "libqhullcpp/QhullError.h"
#include "libqhullcpp/QhullQh.h"
#include "libqhullcpp/QhullFacet.h"
#include "libqhullcpp/QhullFacetList.h"
#include "libqhullcpp/QhullLinkedList.h"
#include "libqhullcpp/QhullVertex.h"
#include "libqhullcpp/QhullPoint.h"
#include "libqhullcpp/QhullVertexSet.h"
// VTK
#include <vtkUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <vtkSmartPointer.h>
#include <vtkCellArray.h>
// other includes...
#include <ostream>
#include <stdexcept>
#include <vector>
#include <set>
#include "ParseCommandLine.h"
#include <tbb/task_scheduler_init.h>
// namespaces
using orgQhull::Qhull;
using orgQhull::RboxPoints;
using orgQhull::QhullError;
using orgQhull::QhullFacet;
using orgQhull::QhullFacetList;
using orgQhull::QhullLinkedList;
using orgQhull::QhullQh;
using orgQhull::QhullVertex;
using orgQhull::QhullVertexSet;
using orgQhull::QhullVertexSetIterator;
using orgQhull::QhullPoint;

float *getPoints(vtkUnstructuredGrid *usp) {
    int npts = usp->GetNumberOfPoints();
    int idx;
    float *point_vector = new float[npts*3]; // point coords vector
    double *dvector;
    for(int i=0;i<npts;i++){
        dvector = usp->GetPoint(i);
        idx=3*i;
        point_vector[idx]   = dvector[0];
        point_vector[idx+1] = dvector[1];
        point_vector[idx+2] = dvector[2];
    }
    return point_vector;
}
// create a gvt mesh object from a vtkUnstructuredGrid
// filename contains the full path to the vtk file.
// tesselation included.
void vtkPointsToGvtMesh(std::string filename,std::string nodename,std::string qhull_control,int domain)
{
    int count;
    float *points;
    //float kd[] = {.96875,.81,.55};
    float kd[] = {.734,.566,.344};
    vtkSmartPointer<vtkUnstructuredGridReader> vtkreader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    vtkUnstructuredGrid *usp;
    api::createMesh(nodename);
    vtkreader->SetFileName(filename.c_str());
    usp = vtkreader->GetOutput();
    vtkreader->Update();
    //count = usp->GetCells()->GetNumberOfCells();
    count = usp->GetNumberOfPoints();
    std::cerr << " read " << count << " cells " << std::endl;
    points = getPoints(usp);
    //usp->GetOrigin(dvector);
    // copy dvector to single
    
    
    api::addMeshVertices(nodename,count,points,true,qhull_control);
    api::addMeshMaterial(nodename,(unsigned)gvt::render::data::primitives::LAMBERT,kd,1.f);  
    api::finishMesh(nodename,true);
}
// and here we go...
int main(int argc, char** argv) {

    std::string dirname("/work/01197/semeraro/stampede2/Projects/GraviT/pygvt");
    std::string filename;
    ParseCommandLine cmd("gvtTess");
    cmd.addoption("control_string", ParseCommandLine::PATH, "qhull control string",1);
    cmd.parse(argc,argv);
    // init gravit and check for validity
    api::gvtInit(argc,argv);
    cntx::rcontext &db = cntx::rcontext::instance();
    int comsize = db.cntx_comm.size;
    int rnk = db.cntx_comm.rank;
    std::cerr << comsize << " ranks " << std::endl;
    tbb::task_scheduler_init *init;
    unsigned int numthreads = 8;
    init = new tbb::task_scheduler_init(numthreads);
    db.getUnique("threads") = numthreads;
    std::string mymeshname,haloname;
    std::string qhull_control;
    int numberofhalos;
    numberofhalos = 25;
    if(cmd.isSet("control_string")) {
        gvt::core::Vector<std::string> ctrl = cmd.getValue<std::string>("control_string");
        qhull_control = ctrl[0];
    } else {
       qhull_control = " " ;
       //qhull_control = "d Qz";
    } 
    // add the halo meshes
    for(int h=0;h<numberofhalos;h++)
    {
        if((h%comsize == rnk) && (h!=11) && (h!=20)) // read this halo and mesh it.
        {
            haloname = "/halo"+std::to_string(h) + ".vtk";
            filename = dirname+haloname;
            mymeshname = "Halo"+std::to_string(h);
            vtkPointsToGvtMesh(filename,mymeshname,qhull_control,h);
        }
    }
    // add the bbox surface
    // back wall
    if(rnk == 0) {
        float kd[] = {1.,1.,1.};
        float kdb[] = {0.5,.695,.726};
        std::vector<float> boxvertex = {0.0,0.0,0.0, 1.0,0.0,0.0, 1.0,1.0,0.0, 0.,1.,0.,
                                    0.,0.,-0.01, 1.0,0.0,-0.01, 1.0,1.0,-0.01, 0.,1.,-0.01};
        std::vector<unsigned> boxfaces = {1,2,3,1,3,4,5,7,6,5,8,7};
        api::createMesh("boxmesh");
        api::addMeshVertices("boxmesh",boxvertex.size()/3,&boxvertex[0]);
        api::addMeshTriangles("boxmesh",boxfaces.size()/3,&boxfaces[0]);
        api::addMeshMaterial("boxmesh",(unsigned)gvt::render::data::primitives::LAMBERT,kdb,1.f);
        api::finishMesh("boxmesh",true);
    }
    db.sync();
    // add instances on rank 0 and sync
    glm::mat4 *m = new glm::mat4(1.f);
    glm::mat4 &mi = (*m);
    float mf[] = { mi[0][0], mi[0][1], mi[0][2], mi[0][3],
                   mi[1][0], mi[1][1], mi[1][2], mi[1][3],
                   mi[2][0], mi[2][1], mi[2][2], mi[2][3],
                   mi[3][0], mi[3][1], mi[3][2], mi[3][3] };
    if(rnk == 0) {
        // halo instances
        for(int h=0;h<numberofhalos;h++) {
            if((h!=11) && (h!=20)){
         mymeshname = "Halo"+std::to_string(h);
         std::string instanceName = "inst"+std::to_string(h);
         api::addInstance(instanceName,mymeshname,mf);
            }
        }
        // add three instances for the back walls of the domain. 
        api::addInstance("rightwall","boxmesh",mf);
        auto roty = new glm::mat4(1.f);
        *roty = glm::rotate(*roty,1.57f,glm::vec3(0.0f,-1.0f,0.0f));
        auto &roti = (*roty);
        float rotyf[] = {roti[0][0],roti[0][1],roti[0][2],roti[0][3],
                     roti[1][0],roti[1][1],roti[1][2],roti[1][3],
                     roti[2][0],roti[2][1],roti[2][2],roti[2][3],
                     roti[3][0],roti[3][1],roti[3][2],roti[3][3]};
        api::addInstance("leftwall","boxmesh",rotyf);
        auto rotz = new glm::mat4(1.f);
        *rotz = glm::rotate(*roty,1.57f,glm::vec3(-1.0f,0.0f,0.0f));
        roti = (*rotz);
        float rotzf[] = {roti[0][0],roti[0][1],roti[0][2],roti[0][3],
                     roti[1][0],roti[1][1],roti[1][2],roti[1][3],
                     roti[2][0],roti[2][1],roti[2][2],roti[2][3],
                     roti[3][0],roti[3][1],roti[3][2],roti[3][3]};
        api::addInstance("floorwall","boxmesh",rotzf);
    }
    db.sync(); // sync the database globally. 
    // camera bits
    auto eye = glm::vec3(3.0,3.0,3.0);
    auto focus = glm::vec3(0.0,0.0,0.0);
    auto upVector = glm::vec3(0.0,1.0,0.0);
    float fov = (float)(30.0*M_PI/180.0);
    int rayMaxDepth = (int)1;
    int raySamples = (int)1;
    float jitterWindowSize = (float)0.5;
    std::string camname = "TCamera";
    std::cout << "addCamera " << std::endl;
    api::addCamera(camname, glm::value_ptr(eye),glm::value_ptr(focus),glm::value_ptr(upVector),fov,rayMaxDepth,raySamples,jitterWindowSize);
    db.sync();
    // a light
    auto lpos = glm::vec3(3.5,3.5,3.5);
    auto lcolor = glm::vec3(3.0,3.0,3.0);
    std::string lightname = "tessLight1";
    //api::addPointLight(lightname,glm::value_ptr(lpos),glm::value_ptr(lcolor));
    lpos = glm::vec3(1.5,3.0,0.5);
    lcolor = glm::vec3(3.0,3.0,4.0);
    lightname = "tessLight2";
    api::addPointLight(lightname,glm::value_ptr(lpos),glm::value_ptr(lcolor));
    db.sync();
    // film bits
    std::string filmname = "TFilm";
     int width = (int)1024;
     int height = (int)1024;
     std::string outputpath = "Tess";
    std::cout << "addFilm " << std::endl;
     api::addFilm(filmname, width, height, outputpath);
     db.sync();
     // rendering bits
     std::string rendername("TScheduler");
     int schedtype;
     int adaptertype;
     // hardwire the domain scheduler for this test. 
     schedtype = gvt::render::scheduler::Domain;
     // embree adapter
     std::string adapter("embree");
#ifdef GVT_RENDER_ADAPTER_EMBREE
     adaptertype = gvt::render::adapter::Embree;
#else
     std::cerr << "Embree adapter missing. Enable embree and recompile" << std::endl;
     exit(1);
#endif
    std::cout << "addRenderer " << std::endl;
     api::addRenderer(rendername, adaptertype, schedtype, camname, filmname);
     db.sync();
     //db.printtreebyrank(std::cout);
     std::cerr << " time to render " << rnk << std::endl;
     api::render(rendername);
     api::writeimage(rendername,"simple");
     MPI_Finalize();
    return 1;
}
