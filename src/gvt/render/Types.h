#ifndef GVT_RENDER_TYPES_H
#define GVT_RENDER_TYPES_H

namespace gvt {
	namespace render {
		namespace adapter {
            /// render engine used
			enum RenderType
            {
                Volume,
                Surface,
                Manta,
                Optix,
                Embree
            };
		} // namespace adapter
        /// schedule used
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
        /// top-level acceleration structure to organize domains within GraviT
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
