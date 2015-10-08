#ifndef GVT_RENDER_TYPES_H
#define GVT_RENDER_TYPES_H

namespace gvt {
	namespace render {
		namespace adapter {
			enum RenderType
            {
                Volume,
                Surface,
                Manta,
                Optix,
                Embree,
                Hybrid
            };
		} // namespace adapter
		namespace scheduler {
			enum ScheduleType
            {
                Image,
                Domain,
                RayWeightedSpread, // PAN: from EGPGV 2012 paper, deprecated, now called LoadOnce
                LoadOnce, // PAN: from TVCG 2013 paper
                LoadAnyOnce, // PAN: from TVCG 2013 paper
                LoadAnother, // PAN: from TVCG 2013 paper
                LoadMany
            };
		} // namespace scheduler
		namespace accelerator {
            enum AccelType
            {
                NoAccel,
                BVH
            };
		}
	} // namespace render
} // namespace gvt
#endif // GVT_RENDER_TYPES_H
