/* ======================================================================================= 
   This file is released as part of GraviT - scalable, platform independent ray tracing
   tacc.github.io/GraviT
   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas at Austin  
   All rights reserved.
   Licensed under the BSD 3-Clause License, (the "License"); you may not use this file     
   except in compliance with the License.                                                  
   A copy of the License is included with this software in the file LICENSE.               
   If your copy does not contain the License, you may obtain a copy of the License at:     
       http://opensource.org/licenses/BSD-3-Clause                                         
   Unless required by applicable law or agreed to in writing, software distributed under   
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY 
   KIND, either express or implied.                                                        
   See the License for the specific language governing permissions and limitations under   
   limitations under the License.
   GraviT is funded in part by the US National Science Foundation under awards ACI-1339863, 
   ACI-1339881 and ACI-1339840
   ======================================================================================= */

// API functions
#include <gvt/core/context/Variant.h>
#include <gvt/render/RenderContext.h>
#include <gvt/render/Schedulers.h>
#include <gvt/render/Types.h>
#include <gvt/render/data/Domains.h>

#include <tbb/task_scheduler_init.h>
#include <thread>

#ifdef GVT_RENDER_ADAPTER_EMBREE
#include <gvt/render/adapter/embree/EmbreeMeshAdapter.h>
#endif

#ifdef GVT_RENDER_ADAPTER_MANTA
#include <gvt/render/adapter/manta/MantaMeshAdapter.h>
#endif

#ifdef GVT_RENDER_ADAPTER_OPTIX
#include <gvt/render/adapter/optix/OptixMeshAdapter.h>
#endif


void gvtInit() {
    // init mpi... or not
    int initialized,rank;
    MPI_Initialized(&initialized);
    if(!initialized)
    {
        MPI_Init(NULL,NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    }
    // initialize the context and a few other things. 
    gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
    if (cntxt == NULL) // context creation failed
    {
        std::cout << "gvtInit: context init failed" << std::endl;
        exit(0);
    }
    else // build base context objects. 
    {
        // some data nodes
        gvt::core::DBNodeH root = cntxt->getRootNode();
        if(rank == 0)
        {
            gvt::core::DBNodeH dataNodes = cntxt->addToSync(cntxt-createNodeFromType("Data","Data",root.UUID()));
            // this node holds instances if any
            cntxt->addToSync(cntxt->createNodeFromType("Instances","Instances",root.UUID()));
        }
        cntxt->syncContext();
        // default camera and light nodes

    }   
}
void addMeshToContext(int loc, Box3D *mshbx, Mesh *mesh,String meshname) {
    // add a mesh to the context.
    // loc is the rank the mesh lives on
    // mshbx is the mesh bounding box
    // mesh is the mesh itself
    
    // grab the context
    gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
    // get the data node.
    gvt::core::DBNodeH root = cntxt->getRootNode();
    gvt::core::DBNodeH dataNodes = root["Data"];
    // meshes get appended to the data node as a child
    gvt::core::DBNodeH amesh=cntxt->createNodeFromType("Mesh",meshname,dataNodes.UUID());
    amesh["file"] = meshname;
    amesh["bbox"] = (unsigned long long)meshbox;
    amesh["ptr"] = (unsigned long long)mesh;
    int rank ;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    gvt::core::DBNodeH loc = cntxt->createNode("rank",rank);
    amesh["Locations"] += loc;
    cntxt->addToSync(amesh);
}
void addInstanceToContext() {}
void addLightToContext() {}
void modifyLight() {}
void addCameraToContext() {}
void addFilmToContext() {}
void setScheduleToContext() {}
void setAdapterInContext() {}
