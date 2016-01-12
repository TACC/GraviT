/* This application tests the OSPRay rendering engine and evaluates some performance
 * aspects of the library. The results can be compared to similar tests of GraviT to 
 * determine the impact of the GraviT overhead. 
 *
 * Tests Performed:
 * 	1) rendering blank screen with no geometry.
 * 	2) rendering large geometry data (Enzo isosurface)
 * */

#include "ospray/ospray.h"
#include <time.h>
#include "timer.h"
#include <iostream>

int main(int argc, const char** argv) {
	// some basic setup
	int width = 1920;
	int height = 1080;
	// timer stuff
	my_timer_t startTime, endTime;
	// empty vertex list
	float vertex[1] ;
	float color[1];
	int32_t index[1];
	ospInit(&argc,argv);
	OSPCamera camera = ospNewCamera("perspective");
	ospSetf(camera, "aspect", width/(float)height);

	osp::vec3f cam_pos = {0.f,0.f,0.f};
	osp::vec3f cam_up = {0.f, 1.f, 0.f};
  osp::vec3f cam_view = {0.1f, 0.f, 1.f};
	ospSetVec3f(camera, "pos", cam_pos);
	ospSetVec3f(camera, "dir", cam_view);
  ospSetVec3f(camera, "up",  cam_up);
  ospCommit(camera);
	//
	// Create empty data set (dont know if this is necessary or not to do empty
	// screen test)
	//
	OSPGeometry mesh = ospNewGeometry("triangles");
	OSPData data = ospNewData(0, OSP_FLOAT3A, vertex);
	ospCommit(data);
	ospSetData(mesh,"vertex",data);

	data - ospNewData(0, OSP_FLOAT4, color);
	ospCommit(data);
	ospSetData(mesh,"vertex.color", data);

	data = ospNewData(0, OSP_INT3, index);
	ospCommit(data);
	ospSetData(mesh,"index",data);

	ospCommit(mesh);

	OSPModel world = ospNewModel();
	ospAddGeometry(world, mesh);
	ospCommit(world);
	// framebuffer and renderer
	OSPRenderer renderer = ospNewRenderer("ao4");
	ospSetObject(renderer, "model", world);
	ospSetObject(renderer, "camera",camera);
	ospCommit(renderer);
	//
	osp::vec2i framebufferdimensions = {width,height};
	OSPFrameBuffer framebuffer = ospNewFrameBuffer(framebufferdimensions,OSP_RGBA_I8,OSP_FB_COLOR | OSP_FB_ACCUM);
	ospFrameBufferClear(framebuffer, OSP_FB_COLOR | OSP_FB_ACCUM);
	
	timeCurrent(&startTime);
	for(int i = 0; i<100;i++)
		ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR| OSP_FB_ACCUM);
	timeCurrent(&endTime);
	double d = timeDifferenceMS(&startTime,&endTime);
	std::cout << "elapsed time " << d << "(ms)" << std::endl;
	return 0;
}
