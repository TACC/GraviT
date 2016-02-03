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
/* This application tests the OSPRay rendering engine and evaluates some performance
 * aspects of the library. The results can be compared to similar tests of GraviT to 
 * determine the impact of the GraviT overhead. 
 *
 * run it like this:
 *   bin/osptest -i /work/01197/semeraro/maverick/DAVEDATA/EnzoPlyData -o spoot -cp -1000.0,0.0,-1000.0 -fov 50.0 -cd 0.,0.0,-1.0 -cu 0.,1.,0. -ld 0,-0.5,-1
 *
 * Tests Performed:
 * 	1) rendering blank screen with no geometry.
 * 	2) rendering large geometry data (Enzo isosurface)
 * */

#include "ospray/ospray.h"
#include <sys/stat.h>
#include <time.h>
#include "timer.h"
#include <iostream>
#include <sstream>
#include <string.h>
#include <ply.h>
#include <glob.h>
#include <vector>

// file writer right out of ospray example code. 
  void writePPM(const char *fileName,
      const int sizeX, const int sizeY,
      const uint32_t *pixel)
  {
    FILE *file = fopen(fileName, "wb");
    fprintf(file, "P6\n%i %i\n255\n", sizeX, sizeY);
    unsigned char *out = (unsigned char *)alloca(3*sizeX);
    for (int y = 0; y < sizeY; y++) {
      const unsigned char *in = (const unsigned char *)&pixel[(sizeY-1-y)*sizeX];
      for (int x = 0; x < sizeX; x++) {
        out[3*x + 0] = in[4*x + 0];
        out[3*x + 1] = in[4*x + 1];
        out[3*x + 2] = in[4*x + 2];
      }
      fwrite(out, 3*sizeX, sizeof(char), file);
    }
    fprintf(file, "\n");
    fclose(file);

    std::string alphaName(fileName);
    alphaName.resize(alphaName.length()-4); // remove ".ppm"
    alphaName.append("_alpha.pgm");

    file = fopen(alphaName.c_str(), "wb");
    fprintf(file, "P5\n%i %i\n255\n", sizeX, sizeY);
    for (int y = 0; y < sizeY; y++) {
      const unsigned char *in = (const unsigned char *)&pixel[(sizeY-1-y)*sizeX];
      for (int x = 0; x < sizeX; x++)
        out[x] = in[4*x + 3];
      fwrite(out, sizeX, sizeof(char), file);
    }
    fprintf(file, "\n");
    fclose(file);
  }
// definitions used in ply file reader
typedef struct Vertex {
	float x,y,z;
	float nx,ny,nz;
	void *other_props;
} Vertex;

typedef struct Face {
	unsigned char nverts;
	int *verts;
	void *other_props;
} Face;

PlyProperty vert_props[] = {
  /* list of property information for a vertex */
  { "x", Float32, Float32, offsetof(Vertex, x), 0, 0, 0, 0 },
  { "y", Float32, Float32, offsetof(Vertex, y), 0, 0, 0, 0 },
  { "z", Float32, Float32, offsetof(Vertex, z), 0, 0, 0, 0 },
  { "nx", Float32, Float32, offsetof(Vertex, nx), 0, 0, 0, 0 },
  { "ny", Float32, Float32, offsetof(Vertex, ny), 0, 0, 0, 0 },
  { "nz", Float32, Float32, offsetof(Vertex, nz), 0, 0, 0, 0 },
};

PlyProperty face_props[] = {
  /* list of property information for a face */
  { "vertex_indices", Int32, Int32, offsetof(Face, verts), 1, Uint8, Uint8, offsetof(Face, nverts) },
};

// determine if file is a directory
bool isdir(const char* path) 
{
	struct stat buf;
	stat(path,&buf);
	return S_ISDIR(buf.st_mode);
}
// determine if a file exists
bool file_exists(const char* path) 
{
	struct stat buf;
	return ( stat(path,&buf) == 0);
}
/*** search a directory for files named *.ply and return a vector containing the full path to
 * each one. 
 **/
std::vector<std::string> findply(const std::string dirname) {
	glob_t result;
	std::string exp = dirname+"/*.ply";
	std::cout << "searching for " << exp << std::endl;
	glob(exp.c_str(),GLOB_TILDE,NULL,&result);
	std::vector<std::string> ret;
	for(int i=0;i<result.gl_pathc;i++){
		ret.push_back(std::string(result.gl_pathv[i]));
	}
	globfree(&result);
	return ret;
}
/*** read a ply file and stuff the data into an ospgeometry object. Expects a triangle
 * mesh. If the ply file contains non triangular faces then bad things will probably happen. 
 * */
void ReadPlyData(std::string filename, float* &vertexarray,float* &colorarray,int32_t* &indexarray,int &nverts, int &nfaces){
	FILE *InputFile;
	PlyFile *in_ply;
	std::string elem_name;
	int elem_count,i,j;
	//int elem_count,nfaces,nverts,i,j;
	int32_t *index;
	Vertex *vert;
	Face *face;
	// default color of vertex
	float color[] = { 0.5f, 0.5f, 1.0f, 1.0f};
	InputFile = fopen(filename.c_str(), "r");
	in_ply = read_ply(InputFile);
	for(i=0;i<in_ply->num_elem_types;i++) {
		elem_name = std::string(setup_element_read_ply(in_ply,i,&elem_count));
		if(elem_name == "vertex") {
			nverts = elem_count;
			vertexarray = (float*)malloc(3*nverts*sizeof(float));//allocate vertex array
			colorarray = (float*)malloc(4*nverts*sizeof(float));//allocate color array
			setup_property_ply(in_ply, &vert_props[0]);
			setup_property_ply(in_ply, &vert_props[1]);
			setup_property_ply(in_ply, &vert_props[2]);
			vert =(Vertex*)malloc(sizeof(Vertex));
			for(j=0;j<elem_count;j++) {
				get_element_ply(in_ply,(void*)vert);
				vertexarray[3*j]   = vert->x;
				vertexarray[3*j+1] = vert->y;
				vertexarray[3*j+2] = vert->z;
				colorarray[4*j]   = color[0];
				colorarray[4*j+1] = color[1];
				colorarray[4*j+2] = color[2];
				colorarray[4*j+3] = color[3];
			}
		} else if (elem_name == "face" ) {
			nfaces = elem_count;
			indexarray = (int32_t*)malloc(3*nfaces*sizeof(int32_t));
			setup_property_ply(in_ply, &face_props[0]);
			face = (Face*)malloc(sizeof(Face));
			for (j=0;j<elem_count;j++) {
				get_element_ply(in_ply, (void*)face);
				indexarray[3*j]   = face->verts[0];
				indexarray[3*j+1] = face->verts[1];
				indexarray[3*j+2] = face->verts[2];
			}
		}
	}
	close_ply(in_ply);
}
void ReadPlyFile(std::string filename, OSPGeometry &mesh ){

	float *vertexarray = NULL;
	float *colorarray = NULL;
	int32_t *indexarray = NULL;
	int nverts = 0;
	int nfaces = 0;
	ReadPlyData(filename,vertexarray,colorarray,indexarray,nverts,nfaces);
#if 0
	FILE *InputFile;
	PlyFile *in_ply;
	std::string elem_name;
	int elem_count,nfaces,nverts,i,j;
	int32_t *index;
	Vertex *vert;
	Face *face;
	// default color of vertex
	float color[] = { 0.5f, 0.5f, 1.0f, 1.0f};
	InputFile = fopen(filename.c_str(), "r");
	in_ply = read_ply(InputFile);
	for(i=0;i<in_ply->num_elem_types;i++) {
		elem_name = std::string(setup_element_read_ply(in_ply,i,&elem_count));
		if(elem_name == "vertex") {
			nverts = elem_count;
			vertexarray = (float*)malloc(3*nverts*sizeof(float));//allocate vertex array
			colorarray = (float*)malloc(4*nverts*sizeof(float));//allocate color array
			setup_property_ply(in_ply, &vert_props[0]);
			setup_property_ply(in_ply, &vert_props[1]);
			setup_property_ply(in_ply, &vert_props[2]);
			vert =(Vertex*)malloc(sizeof(Vertex));
			for(j=0;j<elem_count;j++) {
				get_element_ply(in_ply,(void*)vert);
				vertexarray[3*j]   = vert->x;
				vertexarray[3*j+1] = vert->y;
				vertexarray[3*j+2] = vert->z;
				colorarray[4*j]   = color[0];
				colorarray[4*j+1] = color[1];
				colorarray[4*j+2] = color[2];
				colorarray[4*j+3] = color[3];
			}
		} else if (elem_name == "face" ) {
			nfaces = elem_count;
			indexarray = (int32_t*)malloc(3*nfaces*sizeof(int32_t));
			setup_property_ply(in_ply, &face_props[0]);
			face = (Face*)malloc(sizeof(Face));
			for (j=0;j<elem_count;j++) {
				get_element_ply(in_ply, (void*)face);
				indexarray[3*j]   = face->verts[0];
				indexarray[3*j+1] = face->verts[1];
				indexarray[3*j+2] = face->verts[2];
			}
		}
	}
	close_ply(in_ply);
#endif
	OSPData data = ospNewData(nverts, OSP_FLOAT3, vertexarray);
	ospCommit(data);
	ospSetData(mesh,"vertex",data);

	data = ospNewData(nverts, OSP_FLOAT4, colorarray);
	ospCommit(data);
	ospSetData(mesh,"vertex.color", data);

	data = ospNewData(nfaces, OSP_INT3, indexarray);
	ospCommit(data);
	ospSetData(mesh,"index",data);

	ospCommit(mesh);
}

	
int main(int argc, const char** argv) {
	// default values
	int width = 1920;
	int height = 1080;
	int warmupframes =1;
	int benchframes = 10;
	// timer stuff
	my_timer_t startTime, endTime;
	double rendertime = 0.0;
	double  iotime = 0.0; 
	double modeltime = 0.0;
	// empty vertex list
	float vertex[1] ;
	float color[1];
	int32_t index[1];
	float* vertexarray;
	float* colorarray;
	int32_t* indexarray;
	int nverts,nfaces;
	int numtriangles = 0;
	// file related things
	std::string temp;
	std::string filename;
	std::string filepath("");
	std::string outputfile("");
	// initialize ospray
	ospInit(&argc,argv);
	OSPGeometry mesh;
	OSPModel world = ospNewModel();
	OSPCamera camera = ospNewCamera("perspective");
	// default camera and light settings
	osp::vec3f cam_pos = {-1000.f,0.f,-1000.f};
	osp::vec3f cam_up = {0.f, 1.f, 0.f};
  osp::vec3f cam_view = {0.1f, 0.0f, 0.1f};
	osp::vec3f light_dir = {0.,0.,1.0};
	float cam_fovy = 50.0;
	float lightdirection[] = {0.,0.,1.0};
	// parse the command line
	if( (argc < 2)  ) 
	{
	// no input so render default empty image.
	} 
	else 
	{
	// parse the input
		for(int i = 1;i<argc;i++) 
		{
			const std::string arg = argv[i];
			if(arg == "-i") 
			{ // set the path to the input file
				filepath = argv[++i];
				if (!file_exists(filepath.c_str()))
				{
					std::cout << "File \"" << filepath << "\" does not exist. Exiting." << std::endl;
					return 0;
				// test to see if the file is a directory
				} 
				else if(isdir(filepath.c_str()))
				{ // read all .ply files in a directory
					std::vector<std::string> files = findply(filepath);
					if(!files.empty()) 
					{ // parse the files and add the meshes.
						std::vector<std::string>::const_iterator file;
						for(file=files.begin(); file!=files.end(); file++)
						{
							timeCurrent(&startTime);
							ReadPlyData(*file,vertexarray,colorarray,indexarray,nverts,nfaces);
							timeCurrent(&endTime);
							iotime += timeDifferenceMS(&startTime,&endTime);
							timeCurrent(&startTime);
							mesh = ospNewGeometry("triangles");
							//ReadPlyFile(*file,mesh);
							OSPData data = ospNewData(nverts, OSP_FLOAT3, vertexarray);
							ospCommit(data);
							ospSetData(mesh,"vertex",data);

							data = ospNewData(nverts, OSP_FLOAT4, colorarray);
							ospCommit(data);
							ospSetData(mesh,"vertex.color", data);

							data = ospNewData(nfaces, OSP_INT3, indexarray);
							ospCommit(data);
							ospSetData(mesh,"index",data);

							ospCommit(mesh);
							ospAddGeometry(world,mesh);
							timeCurrent(&endTime);
							modeltime += timeDifferenceMS(&startTime,&endTime);
							numtriangles += nfaces;
						} 
					} 
					else 
					{
						filepath = "";
					}
				} 
				else 
				{ // read a single file into a mesh.
					timeCurrent(&startTime);
					ReadPlyData(filepath,vertexarray,colorarray,indexarray,nverts,nfaces);
					timeCurrent(&endTime);
					iotime += timeDifferenceMS(&startTime,&endTime);
					timeCurrent(&startTime);
					mesh = ospNewGeometry("triangles");
					//ReadPlyFile(filepath,mesh); 
					OSPData data = ospNewData(nverts, OSP_FLOAT3, vertexarray);
					ospCommit(data);
					ospSetData(mesh,"vertex",data);

					data = ospNewData(nverts, OSP_FLOAT4, colorarray);
					ospCommit(data);
					ospSetData(mesh,"vertex.color", data);

					data = ospNewData(nfaces, OSP_INT3, indexarray);
					ospCommit(data);
					ospSetData(mesh,"index",data);

					ospCommit(mesh);
					ospAddGeometry(world, mesh);
					timeCurrent(&endTime);
					modeltime += timeDifferenceMS(&startTime,&endTime);
					numtriangles += nfaces;
				}
			} 
			else if (arg == "-bench") 
			{ // taken from ospray example 
				if (++i < argc)
        {
        	std::string arg2(argv[i]);
          size_t pos = arg2.find("x");
          if (pos != std::string::npos)
					{
          	arg2.replace(pos, 1, " ");
            std::stringstream ss(arg2);
            ss >> warmupframes >> benchframes;
					}
				}
			} 
			else if (arg == "-o") 
			{
				outputfile = argv[++i];
			}
			else if (arg == "-cp")
			{ // set camera position
				if(++i < argc)
				{
					std::string arg2(argv[i]);
					size_t pos = arg2.find(",");
					if(pos != std::string::npos)
					{
						arg2.replace(pos,1," ");
					}
					pos = arg2.find(",");
					if(pos != std::string::npos)
					{
						arg2.replace(pos,1," ");
					}
					float camx,camy,camz;
					std::stringstream ss(arg2);
					ss >> camx >> camy >> camz;
					cam_pos = {camx,camy,camz};
				}
			}
			else if (arg == "-cd")
			{ // set camera direction
				if(++i < argc)
				{
					std::string arg2(argv[i]);
					size_t pos = arg2.find(",");
					if(pos != std::string::npos)
					{
						arg2.replace(pos,1," ");
					}
					pos = arg2.find(",");
					if(pos != std::string::npos)
					{
						arg2.replace(pos,1," ");
					}
					float cdx,cdy,cdz;
					std::stringstream ss(arg2);
					ss >> cdx >> cdy >> cdz;
					cam_view = {cdx,cdy,cdz};
				}
			}
			else if (arg == "-cu")
			{ // set camera up direction
				if(++i < argc)
				{
					std::string arg2(argv[i]);
					size_t pos = arg2.find(",");
					if(pos != std::string::npos)
					{
						arg2.replace(pos,1," ");
					}
					pos = arg2.find(",");
					if(pos != std::string::npos)
					{
						arg2.replace(pos,1," ");
					}
					float cux,cuy,cuz;
					std::stringstream ss(arg2);
					ss >> cux >> cuy >> cuz;
					cam_up = {cux,cuy,cuz};
				}
			}
			else if (arg == "-ld")
			{ // set light direction
				if(++i < argc)
				{
					std::string arg2(argv[i]);
					size_t pos = arg2.find(",");
					if(pos != std::string::npos)
					{
						arg2.replace(pos,1," ");
					}
					pos = arg2.find(",");
					if(pos != std::string::npos)
					{
						arg2.replace(pos,1," ");
					}
					float ldx,ldy,ldz;
					std::stringstream ss(arg2);
					ss >> ldx >> ldy >> ldz;
					light_dir = {ldx,ldy,ldz};
				}
			}
			else if (arg == "-fov")
			{// grab the field of view 
				cam_fovy = atof(argv[++i]);
			}
		}
	}
	//
	// Create empty data set (dont know if this is necessary or not to do empty
	// screen test) if there is no filename given
	//
	timeCurrent(&startTime);
	if(filepath.empty() ){
		std::cout << " empty filepath render blank screen " << std::endl;
		mesh = ospNewGeometry("triangles");
		OSPData data = ospNewData(0, OSP_FLOAT3A, vertex);
		ospCommit(data);
		ospSetData(mesh,"vertex",data);

		data = ospNewData(0, OSP_FLOAT4, color);
		ospCommit(data);
		ospSetData(mesh,"vertex.color", data);

		data = ospNewData(0, OSP_INT3, index);
		ospCommit(data);
		ospSetData(mesh,"index",data);

		ospCommit(mesh);
		ospAddGeometry(world, mesh);
	}
	ospSetVec3f(camera, "pos", cam_pos);
	ospSetf(camera, "aspect", width/(float)height);
	ospSetf(camera, "fovy", cam_fovy);
	ospSetVec3f(camera, "dir", cam_view);
  ospSetVec3f(camera, "up",  cam_up);
  ospCommit(camera);

	ospCommit(world);
	// framebuffer and renderer
	OSPRenderer renderer = ospNewRenderer("obj");
	ospSetObject(renderer, "model", world);
	ospSetObject(renderer, "camera",camera);
	ospCommit(renderer);
	// Light
	OSPLight somelight = ospNewLight(renderer,"DirectionalLight");
	ospSet3f(somelight,"color",1,1,1);
	//ospSet3fv(somelight,"direction",&cam_view.x);
	//ospSet3fv(somelight,"direction",lightdirection);
	//ospSet3f(somelight,"direction",0.,0.,1.);
	ospSetVec3f(somelight,"direction",light_dir);
	ospCommit(somelight);
	OSPData lightArray = ospNewData(1,OSP_OBJECT,&somelight);
	ospSetData(renderer,"lights",lightArray);
	ospCommit(renderer);
	osp::vec2i framebufferdimensions = {width,height};
	OSPFrameBuffer framebuffer = ospNewFrameBuffer(framebufferdimensions,OSP_RGBA_I8,OSP_FB_COLOR | OSP_FB_ACCUM);
	ospFrameBufferClear(framebuffer, OSP_FB_COLOR | OSP_FB_ACCUM);
	timeCurrent(&endTime);
	modeltime += timeDifferenceMS(&startTime,&endTime);
	
	// warmup
	for(int frame = 0; frame <warmupframes;frame++)
		ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR| OSP_FB_ACCUM);
	// benchmark
	timeCurrent(&startTime);
	for(int frame = 0; frame <benchframes;frame++)
		ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR| OSP_FB_ACCUM);
	timeCurrent(&endTime);
	//
	rendertime = timeDifferenceMS(&startTime,&endTime);
	std::cout << numtriangles/1000000. << " million triangles" << std::endl;
	std::cout << "iotime (ms) " << iotime << " modeltime (ms) " << modeltime << std::endl;
	std::cout << rendertime/benchframes << " (ms)/frame " <<  (1000*benchframes)/rendertime << " fps " << std::endl;
	if(!outputfile.empty()) {
		const uint32_t *fb = (uint32_t*)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR );
		writePPM(outputfile.c_str(),width,height,fb);
	}
	return 0;
}
