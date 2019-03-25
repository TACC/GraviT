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

// and here we go...
int main(int argc, char** argv) {

    // qhull library check
    QHULL_LIB_CHECK
    // declare the qhull classes
    RboxPoints rbox;
    Qhull qhull;
    // init gravit and check for validity
    api::gvtInit(argc,argv);
    cntx::rcontext &db = cntx::rcontext::instance();
    int comsize = db.cntx_comm.size;
    int rnk = db.cntx_comm.rank;
    tbb::task_scheduler_init *init;
    unsigned int numthreads = 8;
    init = new tbb::task_scheduler_init(numthreads);
    db.getUnique("threads") = numthreads;
    // see if we have a legal decomposition.
    //std::cerr << "comsize: " << comsize << " rank: " << rnk << std::endl;
    if(comsize > 8) {
        if(rnk == 0)
            std::cerr << " decomposition size > 8 " << std::endl;
        return 0;
    }
    if(comsize & (comsize -1) !=0) { // not power of two comsize
        if(rnk == 0)
            std::cerr << " need 1, 2, 4, or 8 ranks " << std::endl;
        return 0;
    }
    // set up the bounding box for this rank
    // start with global bounding box 
    glm::vec3 bounds_min = {-0.5,-0.5,-0.5};
    glm::vec3 bounds_max = {0.5,0.5,0.5};
    int idx,jdx,kdx;
    // this is the logical rank index
    //  rnk  idx jdx kdx
    //  0     0    0   0
    //  1     1    0   0
    //  2     0    1   0
    //  3     1    1   0
    //  4     0    0   1
    //  5     1    0   1
    //  6     0    1   1
    //  7     1    1   1
    if(comsize > 1) {
        idx = rnk%2;
        jdx = (rnk>>1)%2;
        kdx = (rnk>>2)%2;
        if(idx == 1)
            bounds_min[0] = 0.0;
        else
            bounds_max[0] = 0.0;
        if(comsize > 2) {
            if(jdx == 1)
             bounds_min[1] = 0.0;
            else
             bounds_max[1] = 0.0;
        }
        if(comsize > 4) {
            if(kdx == 1)
             bounds_min[2] = 0.0;
            else 
             bounds_max[2] = 0.0;
        }
    } else {
        idx = 0;
        jdx = 0;
        kdx = 0;
    }
    // set the local bounds
    /*if(idx == 1) 
        bounds_min[0] = 0.0;
    else 
        if(idx == 0)
            bounds_max[0] = 0.0;
    if(jdx == 1)
        bounds_min[1] = 0.0;
    else
        if(jdx == 0)
            bounds_max[1] = 0.0;
    if(kdx == 1)
        bounds_min[2] = 0.0;
    else
        if(kdx == 0) 
            bounds_max[2] = 0.0; */
    // things look good - parse the args
    //if(rnk == 0) {
        std::cout << bounds_min[0] << " " << bounds_max[0] << std::endl;
        std::cout << bounds_min[1] << " " << bounds_max[1] << std::endl;
        std::cout << bounds_min[2] << " " << bounds_max[2] << std::endl;
    //}
    ParseCommandLine cmd("gvtTess");
    cmd.addoption("num_points", ParseCommandLine::INT, "number of points", 1);
    cmd.addoption("control_string", ParseCommandLine::PATH, "qhull control string",1);
    cmd.parse(argc,argv);
    std::string rbox_control;
    std::string qhull_control;
    if(!cmd.isSet("num_points")) {
        rbox_control = "80 D3";
    } else {
        int np;
        np = cmd.get<int>("num_points");
        rbox_control = std::to_string(np) + " D3";
    }
    if(cmd.isSet("control_string")) {
        gvt::core::Vector<std::string> ctrl = cmd.getValue<std::string>("control_string");
        qhull_control = ctrl[0];
    } else {
       qhull_control = " " ;
       //qhull_control = "d Qz";
    } 
    std::cout << " rbox_control: " << rbox_control << std::endl;
    std::cout << " qhull_control: " << qhull_control << std::endl;
    // all ranks create the random point set.
    // Note: each rank creates a possibly different global set of points
    // later each rank will select a subset of it's locally generated
    // set and the union of the points across the ranks will be the
    // actual global set of points. 
    rbox.appendPoints(rbox_control.c_str());
    if(rbox.hasRboxMessage()) {
        std::cerr << "GraviT Tessellation Test error: " << rbox.rboxMessage();
        return rbox.rboxStatus();
    }
    // determine which points are local using the local domain bounds
    int dimension = rbox.dimension();
    int global_count = rbox.count();
    double *global_point_vector = rbox.coordinates();
    int local_count = 0;
    float *local_point_vector;
    double x,y,z;
    std::vector<int> point_index;
    std::cout << " dims " << dimension << " global count " << global_count << std::endl;
    for(int p=0;p<global_count;p++) {
        int pt = p*dimension;
        x = global_point_vector[pt];
        y = global_point_vector[pt+1];
        z = global_point_vector[pt+2];
    //    std::cout << " x " << x << " y " << y << " z " << z << std::endl;
        if((x > bounds_min[0] && x < bounds_max[0]) && 
           (y > bounds_min[1] && y < bounds_max[1]) &&
           (z > bounds_min[2] && z < bounds_max[2])) {
           point_index.push_back(p); // add point index to kept points
           //std::cout << "pushed point "<< p << " to point_index" << std::endl;
        } 
    }
    if(!point_index.empty() && point_index.size() >= 5){// need 5 points to tessellate
        local_count = point_index.size();
        local_point_vector = new float[local_count*dimension];
        int index = 0;
        for(int n : point_index){
            int nt = n*dimension;
            local_point_vector[index]   = global_point_vector[nt];
            local_point_vector[index+1] = global_point_vector[nt+1];
            local_point_vector[index+2] = global_point_vector[nt+2];
            //std::cerr << n << " " << nt << " " << index << std::endl;
            //std::cerr << local_point_vector[index]   << " " 
            //          << local_point_vector[index+1] << " "
            //          << local_point_vector[index+2] << std::endl;
            index += 3;
        }
        // now add the mesh to gravit and let the api do the tessellation. 
        float kd[] = {1.f,1.f,1.f};
        std::string mymeshname = "Tessellation"+std::to_string(rnk);
        api::createMesh(mymeshname);
        api::addMeshVertices(mymeshname,local_count,local_point_vector,true,qhull_control);
        api::addMeshMaterial(mymeshname,(unsigned)gvt::render::data::primitives::LAMBERT,kd,1.f);
        api::finishMesh(mymeshname,true);
        // now have each rank add an instance of the mesh it owns.
        // Each instance needs a transformation matrix to position the
        // particular instance of the mesh. Since the mesh is not being 
        // transformed we pass the identity matrix.
        glm::mat4 *m = new glm::mat4(1.f);
        glm::mat4 &mi = (*m);
        float mf[] = { mi[0][0], mi[0][1], mi[0][2], mi[0][3], 
                       mi[1][0], mi[1][1], mi[1][2], mi[1][3],
                       mi[2][0], mi[2][1], mi[2][2], mi[2][3], 
                       mi[3][0], mi[3][1], mi[3][2], mi[3][3] };
        //std::string instanceTessName = "tessmesh";
        // each rank creates a single unique mesh with a unique instance
        // that points to that mesh
        std::string instanceTessName = mymeshname;
        std::string instanceName = "inst"+std::to_string(rnk);
        api::addInstance(instanceName,instanceTessName,mf);
    }
    db.sync(); // sync the database globally. 
    // now we call qhull to generate a tesslation
    // At this point rbox has created the points. These will be the vertices
    // of the gravit mesh. It remains to tessselate the mesh points.
    // retrieve some info from rbox.
#if 0
    qhull.runQhull("",dimension,global_count,global_point_vector,qhull_control.c_str());
    QhullFacetList facets = qhull.facetList();
    std::cout << "facets contain " << facets.count() << " entities" << std::endl;
    for(QhullFacetList::const_iterator i = facets.begin();i!=facets.end();++i){
        QhullFacet f=*i;
        if(facets.isSelectAll() || f.isGood()) {
            std::cout << " facet id: " << f.id() << std::endl;
            QhullVertexSet vs = f.vertices();
            if(!vs.isEmpty()) {
                std::cout << "this facet has " << vs.count() << " verts" << std::endl;
                for(int k=0;k<vs.count();k++){
                   coordT * points = vs[k].point().coordinates();
                   std::cout << " " << vs[k].point().id();
        //           std::cout << "\n" << points[0] << " " << points[1] << " " << points[2] << std::endl;
                }
                std::cout << std::endl;
            }
        }
    }
#endif
#if 0 
    float *local_point_vector;
    double x,y,z;
    std::vector<int> point_index;
    //if(rnk == 0) {
    //    std::cout << bounds_min[0] << " " << bounds_max[0] << std::endl;
    //    std::cout << bounds_min[1] << " " << bounds_max[1] << std::endl;
    //    std::cout << bounds_min[2] << " " << bounds_max[2] << std::endl;
    //}
    for(int p=0;p<global_count;p++) {
        x = global_point_vector[p];
        y = global_point_vector[p+1];
        z = global_point_vector[p+2];
        //std::cout << " x " << x << " y " << y << " z " << z << std::endl;
        if((x > bounds_min[0] && x < bounds_max[0]) && 
           (y > bounds_min[1] && y < bounds_max[1]) &&
           (z > bounds_min[2] && z < bounds_max[2])) {
           point_index.push_back(p); // add point index to kept points
       //    std::cout << "pushed point "<< p << " to point_index" << std::endl;
        } 
    }
    // add meshes if we have them
    // each rank contributes a different mesh 
    // with a distinct mesh name
    if(!point_index.empty() && point_index.size() >= 5){// need 5 points to tessellate
        local_count = point_index.size();
        local_point_vector = new float[local_count*3];
        int index = 0;
        for(int n : point_index){
            local_point_vector[index]   = global_point_vector[n];
            local_point_vector[index+1] = global_point_vector[n+1];
            local_point_vector[index+2] = global_point_vector[n+2];
            index += 3;
        }
        // now add the mesh to gravit and let the api do the tessellation. 
        float kd[] = {1.f,1.f,1.f};
        std::string mymeshname = "Tessellation"+std::to_string(rnk);
        api::createMesh(mymeshname);
        api::addMeshVertices(mymeshname,local_count,local_point_vector,true);
        api::addMeshMaterial(mymeshname,(unsigned)gvt::render::data::primitives::LAMBERT,kd,1.f);
        api::finishMesh(mymeshname,true);
        // now have each rank add an instance of the mesh it owns.
        // Each instance needs a transformation matrix to position the
        // particular instance of the mesh. Since the mesh is not being 
        // transformed we pass the identity matrix.
        glm::mat4 *m = new glm::mat4(1.f);
        glm::mat4 &mi = (*m);
        float mf[] = { mi[0][0], mi[0][1], mi[0][2], mi[0][3], 
                       mi[1][0], mi[1][1], mi[1][2], mi[1][3],
                       mi[2][0], mi[2][1], mi[2][2], mi[2][3], 
                       mi[3][0], mi[3][1], mi[3][2], mi[3][3] };
        //std::string instanceTessName = "tessmesh";
        // each rank creates a single unique mesh with a unique instance
        // that points to that mesh
        std::string instanceTessName = mymeshname;
        std::string instanceName = "inst"+std::to_string(rnk);
        api::addInstance(instanceName,instanceTessName,mf);
    }
    db.sync(); // sync the database globally. 
#endif
    // now the camera and lights can be set up. 
    // and the scene rendered
    // camera bits
    auto eye = glm::vec3(0.,0.,3.0);
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
    auto lpos = glm::vec3(0.0,1.0,4.0);
    auto lcolor = glm::vec3(2.0,2.0,2.0);
    std::string lightname = "tessLight";
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
//     db.printtreebyrank(std::cout);
     api::render(rendername);
     api::writeimage(rendername,"simple");
     MPI_Finalize();
    return 1;
}
