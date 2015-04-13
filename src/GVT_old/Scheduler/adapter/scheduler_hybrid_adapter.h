/* 
 * File:   scheduler_hybrid_adapter.h
 * Author: jbarbosa
 *
 * Created on February 18, 2014, 1:46 PM
 */

#ifndef SCHEDULER_HYBRID_ADAPTER_H
#define	SCHEDULER_HYBRID_ADAPTER_H

#include "scheduler_adapter.h"

template<class DomainType>
struct scheduler_hybrid_adapter : public scheduler_adapter<DomainType> {
    
    Domain* dom_mailbox;

    scheduler_hybrid_adapter();
    scheduler_hybrid_adapter(const scheduler_hybrid_adapter& orig);
    virtual ~scheduler_hybrid_adapter();


};

#endif	/* SCHEDULER_HYBRID_ADAPTER_H */

