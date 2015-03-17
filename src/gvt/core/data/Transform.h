/* 
 * File:   convert.h
 * Author: jbarbosa
 *
 * Created on March 10, 2014, 12:43 PM
 */

#ifndef GVT_CORE_DATA_TRANSFORM_H
#define	GVT_CORE_DATA_TRANSFORM_H

#include <gvt/core/Debug.h>

/* 
   explicit function template specifications must be defined in the same namespace as the primary template
   rather than put all transforms into gvt::core::data, making primary template a macro that can be included
   at the top of each specialization inside its resepctive namespace
*/

#define GVT_TRANSFORM_TEMPLATE                                                              \
    template<typename SRC_FORMAT, typename DST_FORMAT> struct transform_impl                \
    {                                                                                       \
        static inline DST_FORMAT transform(const SRC_FORMAT& src)                           \
        {                                                                                   \
            GVT_WARNING(DBG_ALWAYS,"CONVERSION NOT IMPLEMENTED : TRYING DYNAMIC CAST");     \
            return dynamic_cast<DST_FORMAT>(src);                                           \
        }                                                                                   \
    };                                                                                      \
                                                                                            \
    template<typename SRC_FORMAT, typename DST_FORMAT>                                      \
    inline DST_FORMAT transform(const SRC_FORMAT& param)                                    \
    {                                                                                       \
        return transform_impl<SRC_FORMAT,DST_FORMAT>::transform(param);                     \
    }                                                                                       

#endif	/* GVT_CORE_DATA_TRANSFORM_H */

