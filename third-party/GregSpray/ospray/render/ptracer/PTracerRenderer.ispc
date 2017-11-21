// ======================================================================== //O
// Copyright 2009-2016 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "ospray/fb/FrameBuffer.ih"
#include "ospray/common/Model.ih"
#include "ospray/common/Ray.ih"
#include "ospray/render/util.ih"
#include "ospray/render/ptracer/PTracerRenderer.ih"
#include "ospray/render/util.ih"
#include "ospray/camera/Camera.ih"
#include "ospray/common/Model.ih"

#include "hits.h"

struct Hit
{
  vec3f point;
  vec3f normal;
  vec3f color;
  float opacity;
  int type;
};


#include "UV.ih"

inline void MyIntersectBox(const Ray& ray,
                         const uniform box3f& box,
                         float& t0,
                         float& t1)
{
  const vec3f mins = mul(sub(box.lower, ray.org), rcp(ray.dir));
  const vec3f maxs = mul(sub(box.upper, ray.org), rcp(ray.dir));

  t0 = max(min(mins.x,maxs.x),
           max(min(mins.y,maxs.y),
               min(mins.z,maxs.z)));

  t1 = min(max(mins.x,maxs.x),
           min(max(mins.y,maxs.y),
               max(mins.z,maxs.z)));
}

inline bool PTracerRenderer_LookForGeometryHit(PTracerRenderer *uniform self, varying Ray &ray, varying float &tTermination)
{
        ray.t = tTermination;
  traceRay(self->super.model, ray);

  if(ray.geomID < 0)
    return false;
        else
        {
                tTermination = ray.t;
                return true;
        }
}

inline void PTracerRenderer_ShadeGeometryHit(PTracerRenderer *uniform self, varying Ray &ray, varying Hit &hit)
{
  DifferentialGeometry dg;
  postIntersect(self->super.model, dg, ray, DG_NG|DG_NS|DG_NORMALIZE|DG_FACEFORWARD|DG_COLOR |DG_MATERIALID|DG_TEXCOORD);
        hit.color   = make_vec3f(dg.color.x, dg.color.y, dg.color.z);
        hit.opacity = dg.color.w;
        hit.point   = dg.P;
        hit.normal  = dg.Ns;
}

inline bool PTracerRenderer_LookForSliceHit(PTracerRenderer *uniform self, varying Ray &ray, varying float& tTermination, uniform int nSlices, uniform vec4f *slices, varying int& id, varying Hit &hit, uniform ExternalRaySOA *uniform raysIn, varying int i)
{
        bool h = false;
        for (uniform int i = 0; i < nSlices; i++)
        {
                vec4f plane = slices[i];
                vec3f pnorm = make_vec3f(plane.x, plane.y, plane.z);
                float d = plane.w;

                float denom = dot(ray.dir, pnorm);

                if (abs(denom) > 0.0001)
                {
                        float t = (d - dot(ray.org, pnorm)) / denom;

                        // print("Slice T: %\n", t);

                        if (t >= ray.t0 && t <= tTermination)
                        {
                                tTermination = t;
                                id = i;
                                h = true;
                        }
                }
        }

        if (h)
        {
                hit.normal = make_vec3f(slices[id].x, slices[id].y, slices[id].z);
                if (dot(hit.normal, ray.dir) > 0)
                {
                        vec3f z = make_vec3f(0.0, 0.0, 0.0);
                        hit.normal = z - hit.normal;
                }
        }
                
        return h;
}

inline void PTracerRenderer_ShadeSliceHit(PTracerRenderer *uniform self, uniform Volume *uniform volume, varying Ray &ray, varying float &sample, varying Hit &hit)
{
        hit.color = volume->transferFunction->getColorForValue(volume->transferFunction, sample);

#if 0
Going to need a separate slice TF
        hit.opacity = volume->transferFunction->getOpacityForValue(volume->transferFunction, sample);
#else
        hit.opacity = 1.0;
#endif
}

inline bool PTracerRenderer_LookForIsoHit(uniform PTracerRenderer *uniform self, varying Ray &ray,
                                                                                                                                                                        varying float sLast, varying float& sThis, 
                                                                                                                                                                        varying float tLast, varying float& tThis,
                                                                                                                                                                        uniform int nIsos, uniform float *uniform isos, varying int& id)
{
        bool hit = false;

        for (int i = 0; i < nIsos; i++)
        {
                float isoval = isos[i];
                if (((isoval >= sLast) && (isoval < sThis)) || ((isoval <= sLast) && (isoval > sThis)))
                {
                        tThis = tLast + ((isoval - sLast) / (sThis - sLast)) * (tThis - tLast);
                        sThis = isoval;
                        id = i;
                        hit = true;
                }
        }
        if (hit)
        {
                vec3f coord = ray.org + tThis * ray.dir;
                // print("ISO HIT\nX %\nY %\nA %\ntThis %\n", coord.x, coord.y, coord.z, tThis);
        }
        return hit;
}

inline void PTracerRenderer_ShadeIsoHit(PTracerRenderer *uniform self, varying Ray &ray, uniform Volume *uniform volume, varying Hit &hit, float value)
{
        vec3f point = ray.org + ray.t * ray.dir;
        hit.normal = safe_normalize(volume->computeGradient(volume, point));
        if (dot(ray.dir, hit.normal) > 0)
                hit.normal = neg(hit.normal);
        hit.color = volume->transferFunction->getColorForValue(volume->transferFunction, value);

#if 0
Going to need a separate isosurface TF
        hit.opacity = volume->transferFunction->getOpacityForValue(volume->transferFunction, value);
#else
        hit.opacity = 1.0;
#endif
}

void PTracerRenderer_traceRays(uniform Renderer *uniform pointer,
    void *uniform perFrameData,
    const uniform int nRaysIn,
    uniform ExternalRaySOA *uniform raysIn)
{
    PTracerRenderer *uniform self = (PTracerRenderer *uniform)pointer;
    uniform Volume *uniform volume = self->super.model->volumes[0];
    uniform float step = volume->samplingStep / volume->samplingRate;
    uniform float epsilon = self->epsilon * step;

    foreach (i = 0 ... nRaysIn)
    {
       enum { NONE, OPAQUE, GEOM, ISO, SLICE, BOUNDARY } term = NONE;
       Ray ray;

       //print("NEW RAY:\n%\n%\n", raysIn->x[i], raysIn->y[i]);

       ray.org = make_vec3f(raysIn->ox[i], raysIn->oy[i], raysIn->oz[i]);
       ray.dir = make_vec3f(raysIn->dx[i], raysIn->dy[i], raysIn->dz[i]);
       if (ray.dir.x == 0.f) ray.dir.x = 1e-6f;
       if (ray.dir.y == 0.f) ray.dir.y = 1e-6f;
       if (ray.dir.z == 0.f) ray.dir.z = 1e-6f;
       ray.geomID = -1;
       ray.primID = -1;
       ray.instID = -1;
       ray.t0  = raysIn->t[i];
       ray.t   = raysIn->tMax[i];
       ray.time = 0.f;
       ray.mask = -1;

       float tTimeout = ray.t;

       vec4f color = make_vec4f(raysIn->r[i], raysIn->g[i], raysIn->b[i], raysIn->o[i]);

       // print("AA\n");

       // tExit is the initial termination of the ray - infinity, or the AO radius
       // Its used to determine if loop termination was due to timeout

       varying float tExit  = ray.t;

       // Get the volume enter/exit t's for the ray irrespective of the rays lifetime
       // tExitVolume is the distance to the exit boundary.  Used to determine if loop
       // termination was due to exitting the volume - which MAY terminate the ray or
       // may cause it to be traced into the next partition of a partitioned dataset.
       // The entry can potentially be negative if the ray origin is inside the volume 

       float tEntry, tExitVolume;
       MyIntersectBox(ray, volume->boundingBox, tEntry, tExitVolume);
       //print("tEntry-tExitVolume\n%\n%\n", tEntry, tExitVolume);

       // print("BB\n");
       // intersectBox(ray, volume->boundingBox, ray.t0, ray.t);
       // If the volume entrypoint is beyong the current ray point, 
       // move the ray point up to the entry point
       if (tEntry < ray.t0) tEntry = ray.t0;
       else if (tEntry > ray.t0) ray.t0 = tEntry;

       // And the loop termination point is intially the ray termination 
       // (if that comes before tExitVolume) or tExitVolume.

       float tTermination  = min(ray.t, tExitVolume);
                
       // The Hit structure is used to retain info if and when we hit a 
       // surface (slice, iso or geometry)

       Hit hit;        // shading info for surface hit
       int hitId = -1; // indx of isosurface or slice that terminates the interval
       
       // Now we see if we hit a slice - this will move tTermination up it we do.
       // print("slice: % % % %\n", volume->slicePlanes[0].x, volume->slicePlanes[0].y, volume->slicePlanes[0].z, volume->slicePlanes[0].w);

       if (PTracerRenderer_LookForSliceHit(self, ray, tTermination, volume->nSlicePlanes, volume->slicePlanes, hitId, hit, raysIn, i)) 
       {
           term = SLICE;
       }

       // Similarly, we look to see if we hit geometry, and again this will move tTermination up it we do.

       if (PTracerRenderer_LookForGeometryHit(self, ray, tTermination))
           term = GEOM;
                                
       // Now iterate up the interval from tEntry to tTzermiantion, accruing 
       // volume contributions and looking for
       // isosurfaces
       varying float tLast, tThis, sThis, sLast; varying iHit = -1;  varying bool opaque = false;
       // Note complicated step - last step to boundary must be included, but step AFTER
       // THAT one must not.
       for (tThis = tEntry; tThis <= tTermination && term != OPAQUE; tThis = (tThis == tEntry) ? (tEntry + epsilon) : (((tThis + step) > tTermination) && (tThis < tTermination)) ? tTermination : tThis + step)
       {
           const vec3f coord = ray.org + tThis * ray.dir;
           //print("---\ntThis %\ntEntry %\nCX %\nCY %\nCZ %\n", tThis, tEntry,coord.x, coord.y, coord.z);
           // print("---\ntThis %\n", tThis);
           if (tThis < 0)
           {
               print("NEG 1!\n");
               break;
           }
           sThis = volume->computeSample(volume, coord);
           //print(" sample %\n",sThis);
// Only look for an isosurface if this is NOT the first iteration.   
// If an isosurface is hit, tThis and sThis will be reset to
// reflect the interpolation within the step.  
// tTermination is then moved up to tThis since we need to bail out of the loop at
// the surface hit.   Note the second condition - 
// this is because of AO and shadow rays cast from an isosurface hit.   Since the
// hit point is linearly interpolated, it might actually be a bit 
// beyond the actual isosurface, so this is a second bulwark against 
// re-hitting the isosurface immediately.
// print("----- LOOP\ntThis %\ntEntry %\ntLast %\nepisilon %\n", tThis, tEntry,tLast,epsilon);
           // if (tThis > tEntry)
               // print("----LOOP  test\ntLast %\nsLast %\ntThis %\nsThis %\n", tLast, sLast, tThis, sThis);
           if (tThis > tEntry && tLast >= epsilon)
           {
          // print("ISO test\ntLast %\nsLast %\ntThis %\nsThis %\n", tLast, sLast, tThis, sThis);
               if (PTracerRenderer_LookForIsoHit(self, ray, sLast, sThis, tLast, tThis, volume->nIsovalues, volume->isovalues, hitId))
                   {
                       term = ISO;
                       tTermination = tThis;
                   }
            // MOVED FROM
               if (tThis < 0)
                 {
                   print("NEG 2!\n");
                 }
       // Add in volume contribution ... average samples at start and end of interval
               if (volume->volume_rendering)
                   {
                       float sVolume = (sLast + sThis) / 2;
                       float sampleOpacity = volume->transferFunction->getOpacityForValue(volume->transferFunction, sVolume);
                       //print("---\nslast %\nsThis %\nsVolume %\nsampleOpacity %\n", sLast,sThis,sVolume, sampleOpacity);
                       if (sampleOpacity > 0)
                       {
                          vec3f sampleColor = volume->transferFunction->getColorForValue(volume->transferFunction, sVolume);
                          vec4f weightedColor = clamp(sampleOpacity / volume->samplingRate) * make_vec4f(sampleColor.x, sampleColor.y, sampleColor.z, 1.0f);
                          color = color + (1.0f - color.w) * weightedColor;
                          //print("sampledcolor\nsamplevolume %\nred %\ngreen %\nblue %\n",sVolume,sampleColor.x,sampleColor.y,sampleColor.z);
                          //print("weightedcolor\nsamplevolume %\nred %\ngreen %\nblue %\n",sVolume,weightedColor.x,weightedColor.y,weightedColor.z);
                          //print("color\nsamplevolume %\nred %\ngreen %\nblue %\n",sVolume,color.x,color.y,color.z);
                          if (min(min(color.x, color.y), color.z) >= 1.0f || color.w > 0.999f)
                          term = OPAQUE;
                       }
                    }
             } //moved this down  from where it says "MOVED FROM" above
                 tLast = tThis;
                 sLast = sThis;
        }
// To get the above loop to terminate correctly at the termination T, the last 
// valid step is ending T is clamped to the termination T and the subsequent 
// step goes PAST there, though its invalid. So if tThis is beyond the termination 
// T we back it up.
       if (tThis > tTermination) 
           tThis = tTermination;

       raysIn->t[i] = tThis;
       ray.t = tThis;

// OK, now if we terminated, it could be for several reasons... If a surface hit 
// happened, shade it depending on what type of hit We only need to shade primary 
// rays - shadow and AO rays are just looking for hits

       if (raysIn->type[i] == EXTERNAL_RAY_PRIMARY)
       {
           if (term == SLICE)
           {
               PTracerRenderer_ShadeSliceHit(self, volume, ray, sThis, hit);
           }
           else if (term == GEOM)
           {
               PTracerRenderer_ShadeGeometryHit(self, ray, hit);
           }
           else if (term == ISO)
           {
               hit.color.y = hit.color.z = 0;
               PTracerRenderer_ShadeIsoHit(self, ray, volume, hit, volume->isovalues[hitId]);
           }

           if (term == GEOM || term == ISO || term == SLICE)
           {
               // Save the sample color for lighting later

               raysIn->sr[i] = hit.color.x;
               raysIn->sg[i] = hit.color.y;
               raysIn->sb[i] = hit.color.z;
               raysIn->so[i] = hit.opacity;
               raysIn->nx[i] = hit.normal.x;
               raysIn->ny[i] = hit.normal.y;
               raysIn->nz[i] = hit.normal.z;
           }
           else
           {
               raysIn->sr[i] = 0;
               raysIn->sg[i] = 0;
               raysIn->sb[i] = 0;
               raysIn->so[i] = 0;
               raysIn->nx[i] = 0;
               raysIn->ny[i] = 0;
               raysIn->nz[i] = 0;
           }

           raysIn->r[i] = color.x;
           raysIn->g[i] = color.y;
           raysIn->b[i] = color.z;
           raysIn->o[i] = color.w;
           //print("color\nred %\ngreen %\nblue %\n",color.x,color.y,color.z);
       }

       raysIn->term[i] = 0;
       if (term == GEOM || term == ISO || term == SLICE)
       {
               // surface lighting model will be computed from surface color
               // when we are spawning lighting-model secondary rays.  We need
               // to factor in the surface opacity here so we can know whether
               // the result is going to be opaque or not.  Its opacity is not
               // affected by lighting.   However, we need the pre-surface
               // accumulated opacity to ramp down the lighting model,
               // so we keep it in so

               float orig_o = raysIn->o[i];
               raysIn->o[i] += (1 - raysIn->o[i])*raysIn->so[i];
               raysIn->so[i] = orig_o;
               
               // print("GEOM: %\n", term == GEOM);
               // print("ISO: %\n", term == ISO);
               // print("SLICE: %\n", term == SLICE);

               raysIn->term[i] |= EXTERNAL_RAY_SURFACE;
       }

       if (term == OPAQUE || raysIn->o[i] > 0.999)
               raysIn->term[i] |= EXTERNAL_RAY_OPAQUE;

       if (tTimeout == tTermination)
               raysIn->term[i] |= EXTERNAL_RAY_TIMEOUT;

       if (ray.t == tExitVolume || term == BOUNDARY)
       {
               raysIn->term[i] |= EXTERNAL_RAY_BOUNDARY;
       }

 //       print("term:\nSURFACE %\nOPAQUE %\nTIMEOUT %\nBOUNDARY %\n", raysIn->term[i] & EXTERNAL_RAY_SURFACE,  raysIn->term[i] & EXTERNAL_RAY_OPAQUE,  raysIn->term[i] & EXTERNAL_RAY_TIMEOUT,  raysIn->term[i] & EXTERNAL_RAY_BOUNDARY);
        }
}

export void PTracerRenderer_setLights(void *uniform _self, uniform vec3f *uniform lts, const uniform uint32 nLts)
{
  // Cast to the actual Renderer subtype.
  uniform PTracerRenderer *uniform self = (uniform PTracerRenderer *uniform)_self;

        if (self->lights) delete[] self->lights;

        self->numLights = nLts;
        if (self->numLights)
        {
                self->lights = uniform new uniform vec3f[self->numLights];
                for (int i = 0; i < self->numLights; i++)
                                self->lights[i] = lts[i];
        }
}

export void PTracerRenderer_setEpsilon(void *uniform _self, uniform float epsilon)
{
  // Cast to the actual Renderer subtype.
  uniform PTracerRenderer *uniform self = (uniform PTracerRenderer *uniform)_self;

        self->epsilon = epsilon;
}

export void PTracerRenderer_setRank(void *uniform _self, uniform int r)
{
  uniform PTracerRenderer *uniform self = (uniform PTracerRenderer *uniform)_self;
        self->rank = r;
}

export void PTracerRenderer_setLightModel(void *uniform _self, uniform int do_shadows, uniform int n_ao_rays, uniform float ao_radius, uniform float Kd, uniform float Ka)
{
  // Cast to the actual Renderer subtype.
  uniform PTracerRenderer *uniform self = (uniform PTracerRenderer *uniform)_self;

        self->do_shadows = do_shadows;
        self->n_ao_rays = n_ao_rays;
        self->ao_radius = ao_radius;
        self->Kd = Kd;
        self->Ka = Ka;
}

void PTracerRenderer_generateAORays(uniform Renderer *uniform pointer,
                                                                                                                                const uniform int nRaysIn,
                                                                                                                                uniform ExternalRaySOA *uniform raysIn,
                                                                                                                                uniform int *uniform offsets,
                                                                                                                                uniform ExternalRaySOA *uniform raysOut)
                                                                                                                                
{
        PTracerRenderer *uniform self = (PTracerRenderer *uniform)pointer;
        uniform Volume *uniform volume = self->super.model->volumes[0];
        uniform float epsilon = self->epsilon * (volume->samplingStep / volume->samplingRate);

        uniform float Ka = self->Ka / self->n_ao_rays;

        for (int i = 0; i < nRaysIn; i++)
        {
                // Only cast secondary rays for primary rays that hit a surface
                if ((raysIn->type[i] == EXTERNAL_RAY_PRIMARY) && (raysIn->term[i] & EXTERNAL_RAY_SURFACE))
                {
                        int t = raysIn->term[i];

                        // Initially, the input ray color is the accumulated result up to the current
                        // surface.   If there is, add in the lighted surface color either by casting
                        // AO and/or shadow rays, or just light the sucker here.  In any event, the
                        // contribution of the surface is attenuated by the current opacity of the primary
                        // ray.

                        // Going to need the surface normal regardless of whether ambient and diffuse are 
                        // added by secondary rays or right here.
                                
                        vec3f surface_normal = make_vec3f(raysIn->nx[i], raysIn->ny[i], raysIn->nz[i]);

                        // Remember, so is the PRE-SURFACE accumulated opacity

                        float ambient_scale = Ka * (1.0 - raysIn->so[i]);

                        float ambient_r = ambient_scale * raysIn->sr[i];
                        float ambient_g = ambient_scale * raysIn->sg[i];
                        float ambient_b = ambient_scale * raysIn->sb[i];

                        vec3f b0 = make_vec3f(1.0f, 0.0f, 0.0f);
                        if (abs(dot(b0, surface_normal)) > 0.95) b0 = make_vec3f(0.0f, 1.0f, 0.0f);
                        vec3f b1 = normalize(cross(b0, surface_normal));
                        b0 = normalize(cross(b1, surface_normal));

                        float ox = raysIn->ox[i] + raysIn->t[i]*raysIn->dx[i];
                        float oy = raysIn->oy[i] + raysIn->t[i]*raysIn->dy[i];
                        float oz = raysIn->oz[i] + raysIn->t[i]*raysIn->dz[i];

                        // print("AO Hit:\nX %\nY %\nZ %\n", ox, oy, oz);
                        // print("SN:\nX %\nY %\nZ %\n", surface_normal.x, surface_normal.y, surface_normal.z);

                        ox = ox + epsilon*surface_normal.x;
                        oy = oy + epsilon*surface_normal.y;
                        oz = oz + epsilon*surface_normal.z;

                        // print("Offset point:\nX %\nY %\nZ %\n", ox, oy, oz);

                        foreach (j = 0 ... self->n_ao_rays)
                        {
                                int offset = offsets[i] + j;
                                int r = ((raysIn->x[i] * 9949 + raysIn->y[i] * 9613 + j*9151)>>8) & 0xff;
                                // int r = j;

                                const float r0 = randomU[r];
                                const float r1 = randomV[r];

                                const float w = sqrt(1.f-r1);
                                const float x = cos((2.f*M_PI)*r0)*w;
                                const float y = sin((2.f*M_PI)*r0)*w;
                                const float z = sqrt(r1)+epsilon;

                                vec3f rd = x*b0 + y*b1 + z*surface_normal;
                                // vec3f rd = surface_normal;
                        
                                const float d = 1.0 / self->n_ao_rays;
                                
                                raysOut->ox[offset]    = ox;
                                raysOut->oy[offset]    = oy;
                                raysOut->oz[offset]    = oz;

                                raysOut->dx[offset]    = rd.x;
                                raysOut->dy[offset]    = rd.y;
                                raysOut->dz[offset]    = rd.z;
                                raysOut->r[offset]     = ambient_r;
                                raysOut->g[offset]     = ambient_g;
                                raysOut->b[offset]     = ambient_b;
                                raysOut->o[offset]     = 0.0;
                                raysOut->t[offset]     = 0.0;
                                raysOut->tMax[offset]  = self->ao_radius;
                                raysOut->x[offset]       = raysIn->x[i];
                                raysOut->y[offset]       = raysIn->y[i];
                                raysOut->type[offset]  = EXTERNAL_RAY_AO;
                                raysOut->term[offset]  = 0;
                        }
                }
  }
}

void PTracerRenderer_ambientLighting(uniform Renderer *uniform pointer,
                                                                                                                                const uniform int nRaysIn,
                                                                                                                                uniform ExternalRaySOA *uniform raysIn)
                                                                                                                                
{
        PTracerRenderer *uniform self = (PTracerRenderer *uniform)pointer;

        uniform float Ka = self->Ka;

        foreach (i = 0 ... nRaysIn)
        {
                if ((raysIn->type[i] == EXTERNAL_RAY_PRIMARY) && (raysIn->term[i] & EXTERNAL_RAY_SURFACE))
                {
                        int t = raysIn->term[i];

                        // Remember, so is the PRE-SURFACE accumulated opacity

                        float ambient_scale = Ka * (1.0 - raysIn->so[i]);

                        raysIn->r[i] += ambient_scale * raysIn->sr[i];
                        raysIn->g[i] += ambient_scale * raysIn->sg[i];
                        raysIn->b[i] += ambient_scale * raysIn->sb[i];
                }
  }
}

void PTracerRenderer_generateShadowRays(uniform Renderer *uniform pointer,
                                                                                                                                const uniform int nRaysIn,
                                                                                                                                uniform ExternalRaySOA *uniform raysIn,
                                                                                                                                uniform int *uniform offsets,
                                                                                                                                uniform ExternalRaySOA *uniform raysOut)
                                                                                                                                
{
        PTracerRenderer *uniform self = (PTracerRenderer *uniform)pointer;
        uniform Volume *uniform volume = self->super.model->volumes[0];
        uniform float epsilon = self->epsilon * (volume->samplingStep / volume->samplingRate);
        uniform float Kd = self->Kd;

        foreach (i = 0 ... nRaysIn)
        {
                // Only cast secondary rays for primary rays that hit a surface

                int offset = offsets[i];
                if (offset != -1)
                {
                        // print("offset: %\n", offset);

                        int t = raysIn->term[i];

                        // If a primary ray hit a surface, then do the lighting and cast whatever secondary rays 
                        // are necessary. 
                        // the block, then propagate it
                        
                        // Initially, the input ray color is the accumulated result up to the current
                        // surface.   If there is, add in the lighted surface color either by casting
                        // AO and/or shadow rays, or just light the sucker here.  In any event, the
                        // contribution of the surface is attenuated by the current opacity of the primary
                        // ray.

                        // Going to need the surface normal regardless of whether ambient and diffuse are 
                        // added by secondary rays or right here.
                                
                        vec3f surface_normal = make_vec3f(raysIn->nx[i], raysIn->ny[i], raysIn->nz[i]);

                        // If there are shadow rays to be cast, then the Kd of the lighting 
                        // model is divided among the cast rays.  Otherwise the Kd * cos(normal, light) times
                        // the surface color is added in to the input ray color.  Again, in either case, 
                        // the contribution is diminished by the opacity of the goo in front.

                        for (int k = 0; k < self->numLights; k++)
                        {
                                // Remember, so is the PRE-SURFACE accumulated opacity
                                float d = dot(surface_normal, self->lights[k]);
                                if (d < 0) d = 0;

                                float dff = (1.0 - raysIn->so[i]) * Kd * d;

                                float dff_r = dff * raysIn->sr[i];
                                float dff_g = dff * raysIn->sg[i];
                                float dff_b = dff * raysIn->sb[i];

                                raysOut->ox[offset]    = raysIn->ox[i] + raysIn->t[i]*raysIn->dx[i] + epsilon*raysIn->nx[i];
                                raysOut->oy[offset]    = raysIn->oy[i] + raysIn->t[i]*raysIn->dy[i] + epsilon*raysIn->ny[i];
                                raysOut->oz[offset]    = raysIn->oz[i] + raysIn->t[i]*raysIn->dz[i] + epsilon*raysIn->nz[i];
                                raysOut->dx[offset]    = self->lights[k].x;
                                raysOut->dy[offset]    = self->lights[k].y;
                                raysOut->dz[offset]    = self->lights[k].z;
                                raysOut->r[offset]     = dff_r;
                                raysOut->g[offset]     = dff_g;
                                raysOut->b[offset]     = dff_b;
                                raysOut->o[offset]     = 0.0;
                                raysOut->t[offset]     = 0.0;
                                raysOut->tMax[offset]  = inf;
                                raysOut->x[offset]       = raysIn->x[i];
                                raysOut->y[offset]       = raysIn->y[i];
                                raysOut->type[offset]  = EXTERNAL_RAY_SHADOW;
                                raysOut->term[offset]  = 0;
                                offset++;
                        }
                }
  }
}

void PTracerRenderer_diffuseLighting(uniform Renderer *uniform pointer,
                                                                                                                                const uniform int nRaysIn,
                                                                                                                                uniform ExternalRaySOA *uniform raysIn)
{
        PTracerRenderer *uniform self = (PTracerRenderer *uniform)pointer;

        uniform float Kd = self->Kd;
        // print("Kd: %\n", Kd);

        foreach (i = 0 ... nRaysIn)
        {
                if ((raysIn->type[i] == EXTERNAL_RAY_PRIMARY) && (raysIn->term[i] & EXTERNAL_RAY_SURFACE))
                {
                        vec3f surface_normal = make_vec3f(raysIn->nx[i], raysIn->ny[i], raysIn->nz[i]);
                        // print("SN:\n%\n%\n%\n", surface_normal.x, surface_normal.y, surface_normal.z);

                        float tot_dff_r = 0;
                        float tot_dff_g = 0;
                        float tot_dff_b = 0;

                        for (int k = 0; k < self->numLights; k++)
                        {
                                // Remember, so is the PRE-SURFACE accumulated opacity
                                float d = dot(surface_normal, self->lights[k]);
                                // print("light:\n%\n%\n%\n", self->lights[k].x, self->lights[k].y, self->lights[k].z);
                                // print("d: %\n", d);

                                if (d > 0)
                                {
                                        float dff = (1.0 - raysIn->so[i]) * dot(surface_normal, self->lights[k]);

                                        float dff_r = dff * raysIn->sr[i];
                                        float dff_g = dff * raysIn->sg[i];
                                        float dff_b = dff * raysIn->sb[i];

                                        tot_dff_r += dff_r;
                                        tot_dff_g += dff_g;
                                        tot_dff_b += dff_b;
                                }
                        }

                        // print("tot diff:\n%\n%\n%\n", tot_dff_r, tot_dff_g, tot_dff_b);
                        // print("incoming opacity %\n", raysIn->so[i]);

                        raysIn->r[i]     = raysIn->r[i] + Kd * (1 - raysIn->so[i]) * tot_dff_r;
                        raysIn->g[i]     = raysIn->g[i] + Kd * (1 - raysIn->so[i]) * tot_dff_g;
                        raysIn->b[i]     = raysIn->b[i] + Kd * (1 - raysIn->so[i]) * tot_dff_b;
                        raysIn->o[i]     = raysIn->o[i] + Kd * (1 - raysIn->so[i]) * raysIn->so[i];
                }
  }
}

export void *uniform PTracerRenderer_createInstance()
{
  // The self object.
  PTracerRenderer *uniform self = uniform new uniform PTracerRenderer;

  // Constructor of the parent class.
  Renderer_Constructor(&self->super, NULL);

  self->super.traceRays                                 = PTracerRenderer_traceRays;
  self->super.generateAORays                    = PTracerRenderer_generateAORays;
  self->super.generateShadowRays        = PTracerRenderer_generateShadowRays;
  self->super.ambientLighting             = PTracerRenderer_ambientLighting;
  self->super.diffuseLighting                   = PTracerRenderer_diffuseLighting;

        self->lights = NULL;

  return self;
}
