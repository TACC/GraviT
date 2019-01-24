/*
 * Tessellation test. Create a random point cloud and create a delaunay tessellation 
 * of it. Using the Qhull library for all of this functionality. Bassically we call the
 * rbox functions to create the point cloud and then use the qhull cpp interface to
 * tessellate it. 
 *
 * This code just tests the cpp interface to qhull libs and does not call 
 * anything related to gravit. This just tests qhull tessellation. There is another
 * application that makes these calls from inside the GraviT api. 
 *
 * See the user_eg3_r.cpp example code from qhull and the qhull cpp code. 
 *
 */

// includes

#include "libqhullcpp/RboxPoints.h"
#include "libqhullcpp/QhullError.h"
#include "libqhullcpp/QhullQh.h"
#include "libqhullcpp/QhullFacet.h"
#include "libqhullcpp/QhullFacetList.h"
#include "libqhullcpp/QhullLinkedList.h"
#include "libqhullcpp/QhullVertex.h"
#include "libqhullcpp/QhullPoint.h"
#include "libqhullcpp/Qhull.h"
#include "libqhullcpp/QhullVertexSet.h"


#include <cstdio>   /* for printf() of help message */
#include <ostream>
#include <stdexcept>

// namespace bits

using orgQhull::Qhull;
using orgQhull::QhullError;
using orgQhull::QhullFacet;
using orgQhull::QhullFacetList;
using orgQhull::QhullLinkedList;
using orgQhull::QhullQh;
using orgQhull::RboxPoints;
using orgQhull::QhullVertex;
using orgQhull::QhullVertexSet;
using orgQhull::QhullVertexSetIterator;
using orgQhull::QhullPoint;

int main(int argc, char** argv) {

    // qhull library check macro... whatever.
    QHULL_LIB_CHECK
    // declare some classes we will use
    RboxPoints rbox;
    Qhull qhull;
    // create 8 points randomly distributed in a
    // unit cube centered at the origin 
    rbox.appendPoints("8 D3");
    if(rbox.hasRboxMessage()) {
        std::cerr << "GraviT Tessellation Test error: " << rbox.rboxMessage();
        return rbox.rboxStatus();
    }
    // call qhull and tessellate the input
    //         control
    // d - delaunay triangulation
    // Qt - triangulate output
    // QJ - add a small perterbation to points to address
    //      coplanar or cospherical input
    // s - output summary
    // i - output verticies for each facet
    // n - output normals with offsets
    // p - output vertex coordinates
    //
    std::string control (" ");
    std::cout << " rbox.comment " << rbox.comment()<< std::endl;
    std::cout << " rbox dimension " << rbox.dimension() << std::endl;
    std::cout << " rbox count " << rbox.count() << std::endl;
    double *pointvector = rbox.coordinates();
    // now dump the points. 
    for(int point =0; point<rbox.count()*3;point=point+3) {
        std::cout << point << " " << pointvector[point]<< " " << pointvector[point+1] << " " << pointvector[point+2] << std::endl;
    } 
    qhull.setOutputStream(&std::cout);
    // try to call the other runQhull function
    qhull.runQhull("",rbox.dimension(),rbox.count(),pointvector,control.c_str());
    //qhull.runQhull(rbox.comment().c_str(),rbox.dimension(),rbox.count(),pointvector,control.c_str());
    QhullFacetList facets = qhull.facetList();
    std::cout << "facets contain " << facets.count() << " entities" << std::endl;
    //std::cout << facets;
    // this section of code demonstrates accessing the facets and vertex data.
    for(QhullFacetList::const_iterator i = facets.begin();i!=facets.end();++i){
        QhullFacet f=*i;
        if(facets.isSelectAll() || f.isGood()) {
            std::cout << " facet id: " << f.id() << std::endl;
            QhullVertexSet vs = f.vertices();
            if(!vs.isEmpty()) {
                std::cout << "this facet has " << vs.count() << " verts" << std::endl;
                // point ids can be accessed like below
                 for(int k=0;k<vs.count();k++)
                    std::cout << " " << vs[k].point().id();
                // or you can use a while loop.
                //QhullVertexSetIterator j = vs;
                //QhullVertex v;
                //QhullPoint p;
                //while(j.hasNext()) {
                //    v = j.next();
                //    p = v.point();
                //    std::cout << " p" << p.id() << "(v" << v.id() << ")";
                //}
                std::cout << std::endl;
            }
            //std::cout << f.printHeader();
        }
    }
   return 1; 
}
