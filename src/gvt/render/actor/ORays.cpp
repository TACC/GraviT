#include <iostream>
#include <sstream>
#include "ORays.h"

#define RAY_PRIMARY              1
#define RAY_SHADOW               2
#define RAY_AO                   3
#define RAY_EMPTY                4

#define RAY_SURFACE              0x1
#define RAY_OPAQUE               0x2
#define RAY_BOUNDARY             0x4
#define RAY_TIMEOUT              0x8

using namespace gvt::render::actor;

ORayList::ORayList(int nrays) 
{
	contents = smem::New(nrays * sizeof(ORay));
}

ORayList::ORayList(SharedP c)
{
	contents = c;
}

void
ORayList::print()
{
	std::stringstream s;
	ORay *r = get();
	for (int i = 0; i < size(); i++, r++)
		s << r->px << " " << r->py << " " << r->x << " " << r->y << " " << r->z << " " << r->dx << " " << r->dy << " " << r->dz << " " << r->t << "\n";
//	Application::GetTheApplication()->Print(s.str());
}
