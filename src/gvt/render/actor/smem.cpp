#include <sstream>
#include "smem.h"
#include <map>

using namespace std;


class knts
{
public:
	knts() { a = 0; d = 0; n = 0; }
	~knts() { 
			// std::cerr << "STATS: " << a << " " << d << " " << (a - d) << "\n";
			// for (auto it = smap.begin(); it != smap.end(); it++)
				// std::cerr << it->first << " " << it->second << "\n";
	}
  void add(void *p) {
		a ++; smap[p] = n++;
	}
	void del(void *p) {d ++; smap.erase(p); }
private:
	int a, d;
  int n;
	map<void *, int> smap;
};

static knts k;


smem::~smem()
{
	// cerr << "smem freed " << sz << " at " << (long)ptr << "\n";
	if (ptr) free(ptr);
	
	k.del((void *)this);
}

static int nalloc = 0;

smem::smem(int n)
{
	if (n > 0)
		ptr = (unsigned char *)malloc(n);
	else
		ptr = NULL;
	sz = n;

	k.add((void *)this);
	// cerr << "smem alloc " << sz << " at " << (long)ptr << "\n";
}
