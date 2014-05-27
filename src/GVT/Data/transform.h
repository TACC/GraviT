/* 
 * File:   convert.h
 * Author: jbarbosa
 *
 * Created on March 10, 2014, 12:43 PM
 */

#ifndef CONVERT_H
#define	CONVERT_H

#include <GVT/common/debug.h>

namespace GVT {
    namespace Data {
        
        template<typename SRC_FORMAT, typename DST_FORMAT> struct transform_impl {
            static inline DST_FORMAT transform(const SRC_FORMAT& src) {
                GVT_WARNING(DBG_ALWAYS,"CONVERSION NOT IMPLEMENTED : TRYING DYNAMIC CAST");
                return dynamic_cast<DST_FORMAT>(src);
            }
        };

        template<typename SRC_FORMAT, typename DST_FORMAT> inline DST_FORMAT transform(const SRC_FORMAT& param) {
            return transform_impl<SRC_FORMAT,DST_FORMAT>::transform(param);
        }

    };
};

#endif	/* CONVERT_H */

