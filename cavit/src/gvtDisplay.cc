#include <iostream>
#include <string>
#include <mpi.h>
#include <sstream>
#include <vector>
#include <boost/timer/timer.hpp>

#include <semaphore.h>  

#include "gvtState.h"
// #include "gvtServer.h"
#include "gvtDisplay.h"

/* include GLUT for display */
#if defined(__APPLE__)
#  include <OpenGL/gl.h>
#  include <GLUT/glut.h>
#  include <ApplicationServices/ApplicationServices.h>
#elif defined(__WIN32__)
#  include <windows.h>
#  include <GL/gl.h>   
#  include <GL/glut.h>
#else
#  include <GL/gl.h>   
#  include <GL/glut.h>
#endif

using namespace std;
using namespace cvt;

static size_t g_numThreads = 0;

size_t display_width = 128;
size_t display_height = 128;
int bench_warmup = 0;
int bench_frames = 0;
int display_numRenderers=0;
void resizeDisplay(int width, int height);

Framebuffer<uchar3> display_framebuffer(display_width, display_height);

Camera g_camera;

static int g_window = 0;

void keyboardFunc(unsigned char k, int, int)
{
}

void specialFunc(int k, int, int)
{
}


static int clickX = 0, clickY = 0;
static bool flip14 = false;

void clickFunc(int button, int state, int x, int y) 
{
}

void motionFunc(int x, int y)
{
}

size_t g_numPix = 0;
size_t g_numPixLimit = 0;
pthread_mutex_t g_numPix_lock;
sem_t g_frameMutex;

void* parallelProcessPixels(void* args)
{
  static MPIBuffer buffer;
    // sem_wait(&g_frameMutex);
  StatePixelsT pixels(0,0,0,0,&display_framebuffer);
  pixels.Recv(MPI_ANY_SOURCE, MPI_COMM_WORLD, buffer);
  sleep(1);
  pthread_mutex_lock(&g_numPix_lock);
  g_numPix += pixels.width*pixels.height;
  if (g_numPix >= g_numPixLimit)
  {
      // sem_post(&g_frameMutex);
  }
  pthread_mutex_unlock(&g_numPix_lock);
}


void displayFunc(void) 
{
  DEBUG("display");
  static         boost::timer::cpu_timer render_timer, bench_timer;
  static    boost::timer::nanosecond_type render_times_accumulated(0);
  static double bench_time=0;
  render_timer.start();
  static MPIBuffer buffer;

  static StateFrame frame;
  frame.width = display_width;
  frame.height = display_height;
    // printf("gvtDisplay: rendering frame %d\n", frame.frame);

  frame.camera = g_camera;
    // printf("bench frames: %d\n", bench_frames);
  if (bench_frames > 0 && frame.frame == bench_warmup)
  {
    bench_timer.start();
    bench_time = MPI_Wtime();
  }
  else if (bench_frames > 0 && frame.frame > bench_frames+bench_warmup)
  {
      // bench_timer.stop();
    double time = boost::chrono::nanoseconds(render_timer.elapsed().wall).count();
    boost::chrono::duration<double> nanoseconds = boost::chrono::nanoseconds(render_timer.elapsed().user + render_timer.elapsed().system);
    boost::chrono::duration<double> seconds = boost::chrono::duration_cast<boost::chrono::seconds>(nanoseconds);
    time = seconds.count();
    time = MPI_Wtime()-bench_time;
    printf("total render time of %d frames: %f avg: %f fps: %f \n",bench_frames, time, time/double(bench_frames),double(bench_frames)/time);
    StateMsg msg;
    msg.msg = "exit";
    msg.SendAll(MPI_COMM_WORLD); 
    MPI_Finalize();
    sleep(5);
    exit(0);
  }
  frame.SendAll(MPI_COMM_WORLD);
  size_t numPix = 0;
  size_t maxPixels = display_width*display_height;
  g_numPixLimit = maxPixels;
  g_numPix = 0;
  while (numPix < maxPixels)
  {
    StatePixelsT pixels(0,0,0,0,&display_framebuffer);
    pixels.Recv(MPI_ANY_SOURCE, MPI_COMM_WORLD, buffer);
    numPix += pixels.width*pixels.height;
      // printf("gvtDisplay numpix: %d\n", numPix);
  }

    // glRasterPos2i(-1, 1);
    // glPixelZoom(1.0f, -1.0f);
  glDrawPixels(display_width,display_height,GL_RGB,GL_UNSIGNED_BYTE,display_framebuffer.GetData());
  glutSwapBuffers();
  frame.frame++;
  render_timer.stop();
  render_times_accumulated += render_timer.elapsed().wall;
  glutPostRedisplay();
  DEBUG("display end");
  if (bench_frames > 0)
    displayFunc();
}

void reshapeFunc(int width, int height) 
{
  resizeDisplay(width,height);
  glViewport(0, 0, width, height);
  display_width = width; display_height = height;
}

void idleFunc() {
  glutPostRedisplay();
}

void initGlut (const std::string name, const size_t width, const size_t height)
{
  display_width = width;
  display_height = height;
  resizeDisplay(display_width,display_height);
  int argc = 0; char** argv = NULL; 
  glutInit(&argc, argv);
  glutInitWindowSize((GLsizei)display_width, (GLsizei)display_height);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowPosition(0, 0);
  g_window = glutCreateWindow(name.c_str());
  glutDisplayFunc(displayFunc);
  glutIdleFunc(idleFunc);
  glutKeyboardFunc(keyboardFunc);
  glutSpecialFunc(specialFunc);
  glutMouseFunc(clickFunc);
  glutMotionFunc(motionFunc);
  glutReshapeFunc(reshapeFunc);
  printf("running glut main\n");
  glutMainLoop();
}

void resizeDisplay(int width, int height)
{
  if (width == display_width && height == display_height)
    return;

  display_width = width;
  display_height = height;
  display_framebuffer.Resize(width,height);
}                           


//
// gvtDisplay
//
void gvtDisplay::Launch(int argc, char** argv)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int commSize;
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  MPI_Comm intercomm;
  MPI_Status status;
  resizeDisplay(width,height);

// char buf[256];
  //   string msg("hello");
  // // if (argc > 1)
  // //   msg = string(argv[1]);
  //   printf("client spawned\n");
  //   MPI_Comm parentcomm;
  //   int errcodes[1];
  //   MPI_Init(&argc, &argv);
  // // boost::mpi::environment env;
  // // boost::mpi::communicator world;
  //   MPI_Comm_get_parent(&parentcomm);

  #ifdef GVT_USE_PORT_COMM
  char port_name[MPI_MAX_PORT_NAME];
  MPI_Comm intercomm;
  MPI_Status status;
    // MPI_Init(&argc, &argv);
  MPI_Lookup_name("gvtDisplay", MPI_INFO_NULL, port_name);
  printf("gvtDisplay connecting to server %s\n", port_name);
  MPI_Comm_connect(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF, &intercomm);
  MPI_Comm globalComm;
    // MPI_Intercomm_merge(intercomm, 0, &globalComm);
  printf("display sending msg\n");
    // MPI_Send((void*)msg.c_str(), msg.length(), MPI_CHAR, 0, 0, intercomm);
  printf("display msg sent\n");
  int uniSize;
  MPI_Comm_size(intercomm, &uniSize);
  printf("gvtDisplay: uni size: %d\n", uniSize);
    #else
  display_numRenderers=commSize-2;
  stateLocal.intercomm = MPI_COMM_WORLD;
    #endif

  bench_frames = 15;
  bench_warmup = 10;
  g_camera.vfov = g_camera.hfov = 35;
  g_camera.eye = glm::vec3(0,0,-100);
  g_camera.lookat = glm::vec3(0,0,0);
  g_camera.up = glm::vec3(0,1,0);

  StateScene scene;
  scene.domains = domains;
  scene.SendAll(stateLocal.intercomm);

  initGlut("gvtDisplay", width,height);
  MPI_Comm_disconnect(&stateLocal.intercomm);
  MPI_Finalize();
}
