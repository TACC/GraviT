/* 
 * File:   HybridScheduleBase.h
 * Author: jbarbosa
 *
 * Created on January 24, 2014, 2:03 PM
 */

#ifndef GVT_RENDER_SCHEDULE_HYBRID_BYRID_SCHEDULE_BASE_H
#define	GVT_RENDER_SCHEDULE_HYBRID_BYRID_SCHEDULE_BASE_H

 namespace gvt {
 	namespace render {
 		namespace schedule {
 			namespace hybrid {
 				struct HybridScheduleBase {

 					int * newMap; 
 					int &size; 
 					int *map_size_buf; 
 					int **map_recv_bufs;
 					int *data_send_buf;

 					HybridScheduleBase(int * newMap, int &size, int *map_size_buf, int **map_recv_bufs, int *data_send_buf) 
 					: newMap(newMap), size(size), map_size_buf(map_size_buf), map_recv_bufs(map_recv_bufs), data_send_buf(data_send_buf) 
 					{}

 					virtual ~HybridScheduleBase() {}
 				private:

 				};

 			}
 		}
 	}
 }

#endif	/* GVT_RENDER_SCHEDULE_HYBRID_BYRID_SCHEDULE_BASE_H */

