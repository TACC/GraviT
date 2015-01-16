/*
 * File:   MantaProcessQueue.h
 * Author: jbarbosa
 *
 * Created on February 28, 2014, 1:45 PM
 */

#ifndef OPTIXPROCESSQUEUE_H
#define OPTIXPROCESSQUEUE_H

#include <Backend/Optix/Domain/OptixDomain.h>
#include <GVT/Backend/MetaProcessQueue.h>

namespace GVT {

namespace Backend {

template <>
void ProcessQueue<GVT::Domain::OptixDomain>::IntersectDomain(
    GVT::Data::ray& ray, GVT::Data::RayVector& newRays);
template <>
void ProcessQueue<GVT::Domain::OptixDomain>::operator()();

}  // namespace Backend;

}  // namespace GVT
#endif /* OPTIXPROCESSQUEUE_H */

