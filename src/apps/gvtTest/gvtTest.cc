#include "gvt.h"

#include <mpi.h>

PLYLoader plyLoader;

void* loadCallback(const gvt::Context& context, gvt::Domain& domain, int LOD)
{
  if (LOD == 0)
    plyLoader.Load"bunny.ply");
  else
    plyLoader.Load("bunny_small.ply");
  domain.setPrimitiveCount(plyLoader.numTriangles);
  domain["vertices"] = plyLoader.vertices;
  domain["vertexIndices"] = plyLoader.vertexIndices;
  domain["normals"] = plyLoader.normals;
  domain["normalIndices"] = plyLoader.normalIndices;
}

void* unLoadCallback(const gvt::Context& context, gvt::Domain& domain, int LOD)
{
  plyLoader.clear();
}

int main(int argc, char** argv)
{
  MPI::Init(&argc, &argv);
  int rank = MPI::COMM_WORLD.Get_rank();
  int size = MPI::COMM_WORLD.Get_size();

  //
  // Initialize gvt
  //
  gvt::Init(argc, argv);

  //
  // Create context
  //
  gvt::Context context = gvt::createContext("triRenderTest");

  //
  // Create a triangle domain
  //
  gvt::Domain tris = context.createDomain("Mr. Bunny");
  {
    tris["transform"] = gvt::AffineTransform().Translate(rank*10,0,0);
    gvt::Program load = context.createProgram("callbackLoader.gvt");
    load["loadCallback"] = &loadCallback();
    load["unLoadCallback"] = &unLoadCallback();
    // Intersction and bounds program define individual primitives
    gvt::Program intersect = context.createProgram("triangleMesh_intersect.gvt");
    gvt::Program bounds = context.createProgram("triangleMesh_bounds.gvt");
    // Hit programs determine shading
    gvt::Program closestHit = context.createProgram("phong_closestHit.gvt");
    gvt::Program anyHit = context.createProgram("phong_anyHit.gvt");
    tris["LODLevels"] = 2;
    tris.setIntersectionProgram(intersect);
    tris.setLoadProgram(load);
    tris.setBoundProgram(bounds);
    tris.setClosestHitProgram(closestHit);
    tris.setAnyHitProgram(anyHit);
    tris["Kd"] = float3(0.3,0.6,0.9);
  }
  context.addDomain(tris);

  if (rank == 0)
  {
    //
    // Camera
    //
    gvt::Program camera = context.createProgram("pinholeCamera.gvt");
    camera["eye"] = float3(0,0,-10*size);
    context->setRayGenerationProgram(camera);

    //
    // Generate image
    //
    context.launch(1920, 1080);

    gvt::Buffer result = context["outputBuffer"];

    //
    //  Render buffer to screen
    //

    // ...
  }

  context.sync();

  MPI::Finalize();
  return 0;
}

}