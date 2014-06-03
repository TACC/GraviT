/* 
 * File:   MantaProcessQueue.h
 * Author: jbarbosa
 *
 * Created on February 28, 2014, 1:45 PM
 */

#ifndef MANTAPROCESSQUEUE_H
#define	MANTAPROCESSQUEUE_H

#include <Backend/Manta/Domain/MantaDomain.h>
#include <GVT/Backend/MetaProcessQueue.h>

namespace GVT {
    namespace Backend {
        template<> void ProcessQueue<GVT::Domain::MantaDomain>::IntersectDomain(GVT::Data::ray& ray, GVT::Data::RayVector& newRays);
        template<> void ProcessQueue<GVT::Domain::MantaDomain>::operator()();
    };
};
#endif	/* MANTAPROCESSQUEUE_H */

