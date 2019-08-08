#pragma once

#include "smem.h"

#define RAY_PRIMARY              1
#define RAY_SHADOW               2
#define RAY_AO                   3
#define RAY_EMPTY                4

#define RAY_SURFACE              0x1
#define RAY_OPAQUE               0x2
#define RAY_BOUNDARY             0x4
#define RAY_TIMEOUT              0x8
#define RAY_EXTERNAL_BOUNDARY		 0x10

namespace gvt {
  namespace render {
    namespace actor {
class ORay
{
public:
	ORay(int px, int py, float x, float y, float z, float dx, float dy, float dz) :
		px(px), py(py), x(x), y(y), z(z), dx(dx), dy(dy), dz(dz), 
		r(0), g(0), b(0), o(0), term(0), type(RAY_PRIMARY) {}

  float x, y, z;
	float dx, dy, dz;
	float r, g, b, o;
	int px, py;
	float t, tMax;
	int type, term;

	void get_current_point(float &cx, float &cy, float &cz) { cx = x + t*dx; cy = y + t*dy; cz = z + t*dz; };
};

class ORayList 
{
public:
	ORayList(int nrays);
	ORayList(SharedP contents);

	int size() { return contents->get_size() / sizeof(ORay); }
	ORay *get() { return (ORay *)contents->get(); }
	SharedP get_ptr() { return contents; };
	void print();

private:
	SharedP contents;
};
}
}
}
