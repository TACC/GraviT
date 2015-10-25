
// begin Manta includes
#include <Core/Exceptions/Exception.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Geometry/BBox.h>
#include <Interface/Context.h>
#include <Interface/LightSet.h>
#include <Interface/MantaInterface.h>
#include <Interface/Object.h>
#include <Interface/Scene.h>
#include <Model/Groups/DynBVH.h>
#include <Model/Groups/Mesh.h>
#include <Model/Lights/PointLight.h>
#include <Model/Materials/Phong.h>
#include <Model/Materials/Lambertian.h>
#include <Model/Primitives/KenslerShirleyTriangle.h>
#include <Model/Readers/PlyReader.h>
// end Manta includes

// Manta includes
#include <Core/Math/CheapRNG.h>
#include <Core/Math/MT_RNG.h>
#include <Engine/Shadows/HardShadows.h>
#include <Engine/Shadows/NoShadows.h>
#include <Model/AmbientLights/AmbientOcclusionBackground.h>
#include <Model/AmbientLights/ArcAmbient.h>
#include <Model/AmbientLights/ConstantAmbient.h>
#include <Model/AmbientLights/EyeAmbient.h>
#include <Model/Primitives/Cube.h>
#include <Model/Primitives/Sphere.h>
// end Manta includes

//manta includes

#include <Interface/TexCoordMapper.h>
#include <Core/Geometry/AffineTransform.h>
#include <Model/Instances/InstanceMaterial.h>
#include <Model/Instances/Instance.h>
#include <Model/Groups/ObjGroup.h>
#include "gvtMCube.h"
// end Manta includes

    #include <iostream>
    #include <string>
    #include <mpi.h>
    #include <pthread.h>
    #include <vector>
    #include <sstream>
    #include <stack>

    #include <boost/timer/timer.hpp>

    #include "gvtState.h"
    #include "gvtServer.h"
    #include "gvtWorker.h"
using namespace std;
using namespace cvt;

struct MantaDomain
{
  MantaDomain()
  {
  }
  void SetDomain(StateDomain* dom) {domain=dom;}
  void Init()
  {
    mesh = new Manta::Mesh();
    accel = new Manta::DynBVH();
    accel->setGroup(mesh);
    t.initWithIdentity();
    t.translate(Manta::Vector(0,0,0));
    i = new Manta::Instance(accel, t);
  }
  StateDomain* domain;
  Manta::AffineTransform t;
  Manta::Instance* i;
  Manta::AccelerationStructure* accel;
  Manta::Mesh* mesh;
};

struct MantaContext
{
  MantaContext()
  {

    rtrt = Manta::createManta();

    Manta::Material *material = new Manta::Lambertian(Manta::Color(Manta::RGBColor(0.f, 0.f, 0.f)));
    Manta::MeshTriangle::TriangleType triangleType = Manta::MeshTriangle::KENSLER_SHIRLEY_TRI;

    // Create BVH 
    // Manta::DynBVH* as = new Manta::DynBVH();
    // as->setGroup(meshManta);



    //Create light set
    Manta::LightSet* lights = new Manta::LightSet();
    // lights->add(new Manta::PointLight(Manta::Vector(0, -5, 8), Manta::Color(Manta::RGBColor(1, 1, 1))));

    // Create ambient light
    Manta::AmbientLight* ambient;
    ambient = new Manta::AmbientOcclusionBackground(Manta::Color::white()*0.5, 1, 36);


    world = new Manta::Group();
    Manta::Cube* box = new Manta::Cube(material,Manta::Vector(0,0,0), Manta::Vector(100,10,100));
    // world->add(box);

    //Create context
    // Manta::PreprocessContext context(rtrt, 0, 1, lights);
    pContext = new Manta::PreprocessContext(rtrt, 0, 1, lights);
    material->preprocess(*pContext);
    domainAccel = new Manta::DynBVH(/* debug msgs */);
    world->preprocess(*pContext);
    domainAccel->setGroup(world);
   // domainAccel->preprocess(*pContext);
    // as->preprocess(context);

    //Select algorithm
    Manta::ShadowAlgorithm* shadows;
    shadows = new Manta::NoShadows();
    Manta::Scene* scene = new Manta::Scene();


    scene->setLights(lights);
    scene->setObject(domainAccel);
    Manta::RandomNumberGenerator* rng = NULL;
    Manta::CheapRNG::create(rng);

    rContext = new Manta::RenderContext(rtrt, 0, 0/*proc*/, 1/*workersAnimandImage*/,
            0/*animframestate*/,
            0/*loadbalancer*/, 0/*pixelsampler*/, 0/*renderer*/, shadows/*shadowAlgorithm*/, 0/*camera*/, scene/*scene*/, 0/*thread_storage*/, rng/*rngs*/, 0/*samplegenerator*/);


    bunnyObj = new Manta::ObjGroup("/work/01336/carson/data/bunny.obj", material, Manta::MeshTriangle::KENSLER_SHIRLEY_TRI);
    bunnyAS = new Manta::DynBVH();
    bunnyObj->preprocess(*pContext);
    bunnyAS->setGroup(bunnyObj);
    bunnyAS->preprocess(*pContext);
  // Manta::LightSet* lights = new Manta::LightSet();
  // assert(MantaRenderer::singleton()->getEngine());
  }
  void AddDomain(StateDomain domain)
  {
    Manta::Material *material = new Manta::Lambertian(Manta::Color(Manta::RGBColor(domain.id%255, (domain.id+128)%255, (domain.id+200)%255)));
    Manta::Cube* box = new Manta::Cube(material,
      Manta::Vector(domain.bound_min[0], domain.bound_min[1], domain.bound_min[2]),
      Manta::Vector(domain.bound_max[0], domain.bound_max[1], domain.bound_max[2]));
    Manta::Sphere* sphere = new Manta::Sphere(material,
      Manta::Vector(domain.bound_min[0], domain.bound_min[1], domain.bound_min[2]), (domain.bound_max[0]-domain.bound_min[0])/2.0);
    // sphere->computeBounds(*pContext, 0,1);
    Manta::Vector dmin(domain.bound_min[0], domain.bound_min[1], domain.bound_min[2]);
    Manta::Vector dmax(domain.bound_max[0], domain.bound_max[1], domain.bound_max[2]);
    Manta::Vector drange = dmax-dmin;
    Manta::AffineTransform t;
    t.initWithIdentity();
    t.scale(drange*2);
    // t.scale(Manta::Vector(100,100,100));
    t.translate(dmin+drange/2.0);
    // t.translate(Manta::Vector(3,0,0));
    Manta::Instance* instance = new Manta::Instance(bunnyAS, t);
    instance->preprocess(*(pContext));
    Manta::gvtMCube* domBox = new Manta::gvtMCube(material,
      Manta::Vector(domain.bound_min[0], domain.bound_min[1], domain.bound_min[2]),
      Manta::Vector(domain.bound_max[0], domain.bound_max[1], domain.bound_max[2]), domain.id, instance);
    domBox->preprocess(*(pContext));
    
    world->add(domBox);
  }
  void Update()
  {              
    world->computeBounds(*pContext, 0,1);
    world->preprocess(*(pContext));
    domainAccel->setGroup(world);
    // domainAccel->computeBounds(*pContext, 0,1);
    domainAccel->preprocess(*(pContext));
  }
  Manta::PreprocessContext* pContext;
  Manta::RenderContext* rContext;
  Manta::MantaInterface* rtrt;
  Manta::Group* world;
  Manta::AccelerationStructure* domainAccel;
  Manta::ObjGroup* bunnyObj;
  Manta::DynBVH* bunnyAS;
};


std::vector<pthread_t> threads;
std::vector<StateTile> work_pool;
std::vector<StateTile> finished_work;
int thread_counter=0;
pthread_mutex_t thread_counter_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t work_pool_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t finished_work_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t work_wait = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mpi_lock = PTHREAD_MUTEX_INITIALIZER;
MantaContext* mContext;
int numWorkWaiting = 0;

Camera camera;
StateFrame frame;
StateScene scene;

void intersectDomains(Manta::RayPacket& mRays, unsigned int* pixIds, FramebufferT* framebuffer)
{
  mRays.resetHits();

  mContext->domainAccel->intersect(*(mContext->rContext), mRays);

  for(int i=mRays.begin(); i < mRays.end(); i++)
  {
    if (mRays.wasHit(i))
    {
      unsigned int id = mRays.getScratchpad<unsigned int>(0)[i];
        // uchar3 color = {0,0,255};
      uchar3 color = {(id)%255,(id+128)%255,(id+200)%255};
      framebuffer->data[pixIds[i]] = color;
        //successful hit
    } 
  }
}

void* worker_thread(void*)
{
  int threadId =0;

  pthread_mutex_lock(&thread_counter_lock);
  threadId = thread_counter++;
  pthread_mutex_unlock(&thread_counter_lock);
  StateTile work;

  while (1)
  {
    bool gotWork=false;        
    Manta::RayPacketData rpData;
        // Manta::RenderContext renderContext;
    Manta::RayPacket mRays(rpData, Manta::RayPacket::UnknownShape, 0, 64, 0, Manta::RayPacket::NormalizedDirections);
    unsigned int pixIds[64];
    pthread_mutex_lock(&work_pool_lock);
    if (!work_pool.empty())
    {
      gotWork=true; 
      work = work_pool.back();
      work_pool.pop_back();
    }
    pthread_mutex_unlock(&work_pool_lock);
    if (gotWork)
    {
      Manta::Ray ray;
      const glm::vec3& gu = camera.u;
      const glm::vec3& gv = camera.v;
      Manta::Vector u(gu[0],gu[1],gu[2]);
      Manta::Vector v(gv[0],gv[1],gv[2]);
      Manta::Vector eye(camera.eye[0], camera.eye[1], camera.eye[2]);
      Manta::Vector direction(camera.direction[0], camera.direction[1], camera.direction[2]);
        // printf("frame width height: %d %d\n", frame.width, frame.height);
      int rayCounter = 0;

      for(int x = work.x; x < work.x +work.width;x++)
      {
        for(int y=work.y; y< work.y+work.height;y++)
        {
          const float u_image = float(x)/float(frame.width)-.5;
          const float v_image = float(y)/float(frame.height)-.5;         
          Manta::Vector dir(v*v_image+u*u_image+direction);
          dir.normalize();
          pixIds[rayCounter] = x+y*frame.width;
          mRays.setRay(rayCounter, eye, dir);
          mRays.setFlag(Manta::RayPacket::ConstantOrigin);
          uchar3 color = {(dir[0]/2.0+.5)*255,(dir[1]/2.0+.5)*255,(dir[2]/2.0+.5)*255};          
          work.framebuffer->data[pixIds[rayCounter]] = color;
          rayCounter++;
          if (rayCounter == 64)
          {
              //packet complete, trace rays
            rayCounter = 0;

            intersectDomains(mRays, pixIds, work.framebuffer);
          }
            // minU = Min(minU, u_image);
            // maxU = Max(maxU, u_image);
        }
      }
      if (rayCounter != 0)
      {
          // trace rays
        Manta::RayPacket subPacket(mRays, 0, rayCounter);
        intersectDomains(subPacket, pixIds, work.framebuffer);
      }


      pthread_mutex_lock(&finished_work_lock);
      finished_work.push_back(work);
        // printf("thread completed work %d of %d\n", finished_work.size(), numWorkWaiting);
      if (finished_work.size() == numWorkWaiting)
      {
          // printf("worker threads completed, unlocking work wait\n");
        pthread_mutex_unlock(&work_wait);
      }
      pthread_mutex_unlock(&finished_work_lock);
    }
  }
}

    //
    // server
    //
    // int main(int argc, char** argv)
void Worker::Launch(int argc, char** argv)
{
  mContext = new MantaContext();
  int rank, size;

  MPI_Comm intercomm;
  MPI_Status status;
      #ifdef GVT_USE_PORT_COMM
        // MPI_Init(&argc, &argv);
  char port_name[MPI_MAX_PORT_NAME];
  char conName[MPI_MAX_PORT_NAME];
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Comm_size(comm, &size);
  sprintf(conName, "gvtRenderer%d", rank-1);
  printf("client %d connecting\n", rank);
  MPI_Lookup_name(conName, MPI_INFO_NULL, port_name);
  MPI_Comm_connect(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF, &stateLocal.intercomm);
        // MPI_Intercomm_merge(stateLocal.intercomm, 0, &intercomm);
        // MPI_Comm_rank(stateLocal.intercomm, &rank);
           //MPI_Comm_connect(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF, &intercomm);
  printf("client %d connected\n", rank);
        #else
  stateLocal.intercomm = MPI_COMM_WORLD;
        #endif

  threads.resize(19);
  for(int i =0; i < threads.size();i++)
  {
    pthread_create(&threads[i],NULL,&worker_thread, NULL);
  }

  // int size;
  {
    stateUniversal.Recv(MPI_ANY_SOURCE, stateLocal.intercomm, buffer);
  }

  bool done = false;
  StateMsg msg;
  camera = frame.camera;
  Framebuffer<uchar3> framebuffer;
  StateTile tile;

  double times_render=0;
  boost::timer::cpu_timer render_timer, pixel_timer;
  render_timer.stop();
  pixel_timer.stop();
  boost::timer::nanosecond_type render_times_accumulated(0);
  boost::timer::nanosecond_type pixel_times_accumulated(0);
  std::vector<StatePixelsT*> pixels_sent;
  pixels_sent.resize(256);
  int pixels_sent_c=0;

  gvtContext context;
  context.mpi_rank = rank;
  context.mpi_size = size;

  while(!done)
  {
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    // printf("worker got probe, tag: %d\n", status.MPI_TAG);
    if (status.MPI_TAG == msg.tag)
    {
      done = true;
    }
    if (status.MPI_TAG == scene.tag)
    {
      scene.Recv(MPI_ANY_SOURCE,stateLocal.intercomm, buffer);
      printf("worker got scene\n");
      for(int i =0; i < scene.domains.size(); i++)
      {
        printf("domain id: %zu minBound %f %f %f \n", scene.domains[i].id, 
          scene.domains[i].bound_min[0],scene.domains[i].bound_min[1],scene.domains[i].bound_min[2]);
        mContext->AddDomain(scene.domains[i]);
      }
      mContext->Update();
    }
    if (status.MPI_TAG == frame.tag)
    {
        //get frame
      frame.Recv(MPI_ANY_SOURCE, MPI_COMM_WORLD, buffer);
      framebuffer.Resize(frame.width, frame.height);
      camera = frame.camera;
      camera.Update();

      StateRequest workRequest(GVT_WORK_REQUEST);
      workRequest.Send(1, stateLocal.intercomm);

    }
    if (status.MPI_TAG == tile.tag)
    {
      size = 0;
      tile.Recv(MPI_ANY_SOURCE, stateLocal.intercomm, buffer);
      tile.framebuffer = &framebuffer;
              // printf("gvtRenderer %d: tile recieved\n", rank);
            // printf("rank %d recieved tile %d %d %d %d\n", rank, tile.x, tile.y, tile.width, tile.height);
      if (tile.width <= 0 || tile.height <= 0)
        continue;
//render
      LoadBalancer2D<StateTile> loadBalancer = LoadBalancer2D<StateTile>(tile.x,tile.y,tile.width,tile.height,threads.size());
      work_pool.resize(0);
      finished_work.resize(0);
      pthread_mutex_lock(&work_pool_lock);
      pthread_mutex_unlock(&work_wait);
      StateTile work = loadBalancer.Next();
      while(work.width)
      {
        work.framebuffer = &framebuffer;
        work_pool.push_back(work);
        work = loadBalancer.Next();
      }
      numWorkWaiting = work_pool.size();
      if (numWorkWaiting > 0)
        pthread_mutex_lock(&work_wait);
      pthread_mutex_unlock(&work_pool_lock);
      pthread_mutex_lock(&work_wait);  //wait for work to complete
          //end render
      pixel_timer.start();
      tile.Run(context);
      pixel_times_accumulated += pixel_timer.elapsed().wall;
      pixel_timer.stop();
      StateRequest workRequest(GVT_WORK_REQUEST);
      workRequest.Send(1, stateLocal.intercomm);
    }
  }
}
