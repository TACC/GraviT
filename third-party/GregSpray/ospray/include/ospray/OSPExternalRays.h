#pragma once

#define OSP_WITH_EXTERNAL_API 1

#define EXTERNAL_RAY_PRIMARY 	1
#define EXTERNAL_RAY_SHADOW 	2
#define EXTERNAL_RAY_AO 	3
#define EXTERNAL_RAY_EMPTY	4

#define EXTERNAL_RAY_SURFACE	0x1
#define EXTERNAL_RAY_OPAQUE	0x2
#define EXTERNAL_RAY_BOUNDARY	0x4
#define EXTERNAL_RAY_TIMEOUT	0x8

#if defined(__cplusplus)
namespace xif
{
#endif

  struct ExternalRaySOA
  {
    float *ox;
    float *oy;
    float *oz;
    float *dx;
    float *dy;
    float *dz;
    float *nx;
    float *ny;
    float *nz;
	float *sample;
    float *r;	// accumulated color in front of surface, if any
    float *g;
    float *b;
    float *o;
    float *sr;	// unlit surface color, unoccluded
    float *sg;
    float *sb;
    float *so;
    float *t;
    float *tMax;
    int   *x;
    int   *y;
    int   *type;
    int   *term;
  };

#if defined(__cplusplus)
};
#endif

#if defined(__cplusplus)
#include <stdlib.h>

namespace osp
{

	class ExternalRays
	{
	public:
		ExternalRays() {count = 0; ptr = NULL; }
		ExternalRays(void *p, int n) : ptr(p), count(n) {}

		void Allocate(int n)
		{
			count = n;
			if (n == 0)
				ptr = NULL;
			else
			{
				// int nn = (n + 64) & 0xfffffffc;
				// ptr   = malloc(nn*(19*sizeof(float) + 4*sizeof(int)));

				int nn = n;
				ptr   = malloc(nn*(19*sizeof(float) + 4*sizeof(int)));

				xr.ox   = (float *)ptr;
				xr.oy   = xr.ox + nn;
				xr.oz   = xr.oy + nn;
				xr.dx   = xr.oz + nn;
				xr.dy   = xr.dx + nn;
				xr.dz   = xr.dy + nn;
				xr.nx   = xr.dz + nn;
				xr.ny   = xr.nx + nn;
				xr.nz   = xr.ny + nn;
				xr.r    = xr.nz + nn;
				xr.g    = xr.r + nn;
				xr.b    = xr.g + nn;
				xr.o    = xr.b + nn;
				xr.sr   = xr.o + nn;
				xr.sg   = xr.sr + nn;
				xr.sb   = xr.sg + nn;
				xr.so   = xr.sb + nn;
				xr.t    = xr.so + nn;
				xr.tMax = xr.t + nn;
				xr.x    = (int *)(xr.tMax + nn);
				xr.y    = xr.x + nn;
				xr.type = xr.y + nn;
				xr.term = xr.type + nn;
			}
		};

		~ExternalRays()
		{ 
			free(ptr);
		}

		int GetCount() { return count; }
		void *GetPtr() { return ptr; }
		struct xif::ExternalRaySOA *GetRays() { return &xr; }

		void Trim(int n) { count = n; }

		struct xif::ExternalRaySOA xr;
		void  *ptr;
		int   count;
	};

};

typedef osp::ExternalRays *OSPExternalRays;
#endif

