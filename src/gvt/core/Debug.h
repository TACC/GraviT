#ifndef GVT_CORE_DEBUG_H
#define GVT_CORE_DEBUG_H

#ifdef GVT_DEBUG
#define DEBUG_CERR(x) std::cerr << (x) << std::endl;
#define DEBUG(x) x
#else
#define DEBUG_CERR(x)
#define DEBUG(x)
#endif // GVT_DEBUG

#endif // GVT_CORE_DEBUG_H