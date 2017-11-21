// ======================================================================== //
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

// ospray
#include "ospray/lights/Light.h"
#include "ospray/common/Data.h"
#include "ospray/common/Core.h"
#include "ospray/include/ospray/OSPExternalRays.h"

// #include "ospray/common/parallel_for.h"
#include "ospray/render/ptracer/PTracerRenderer.h"

#include "hits.h"

// ispc exports
#include "Renderer_ispc.h"
#include "PTracerRenderer_ispc.h"

namespace ospray {

  void PTracerRenderer::commit()
  {
    // Create the equivalent ISPC PTracerRenderer object.
    if (ispcEquivalent == NULL) {
      ispcEquivalent = ispc::PTracerRenderer_createInstance();
		}

		do_shadows = getParam1i("do_shadows", 0);
		n_ao_rays = getParam1i("n_ao_rays", 16);
		ao_radius = getParam1f("ao_radius", 1.0);
		Kd = getParam1f("Kd", 0.4);
		Ka = getParam1f("Ka", 0.6);
		rank = getParam1i("rank", 0);
		epsilon = getParam1f("epsilon", 0.02);

	  ispc::PTracerRenderer_setEpsilon(ispcEquivalent, epsilon);
	  ispc::PTracerRenderer_setLightModel(ispcEquivalent, do_shadows, n_ao_rays, ao_radius, Kd, Ka);
	  ispc::PTracerRenderer_setRank(ispcEquivalent, rank);

		// Set the lights if any.
		Data *lightsData = (Data *)getParamData("lights", NULL);
		if (lightsData)
		{
			numLights = lightsData->size();
			ispc::PTracerRenderer_setLights(ispcEquivalent, (ispc::vec3f *)lightsData->data, lightsData->size());
		}
		else
		{
			numLights = 0;
			ispc::PTracerRenderer_setLights(ispcEquivalent, NULL, 0);
		}

		// Initialize state in the parent class, must be called after the ISPC
		// object is created.
		Renderer::commit();
  }

	OSPExternalRays PTracerRenderer::traceExternalRays(OSPExternalRays raysIn)
	{
		ispc::Renderer_traceRays(getIE(), NULL, raysIn->GetCount(), (struct ispc::ExternalRaySOA *)raysIn->GetRays());

		int *ao_offsets = new int[raysIn->GetCount()];
		int *shadow_offsets = new int[raysIn->GetCount()];

		// Figure out how much space must be available in output ray buffer for each input
		// ray based on the state of that ray after tracing it through the local volume.  

		xif::ExternalRaySOA *rays = (xif::ExternalRaySOA *)raysIn->GetRays();

		int nOutputRays = 0; 

		// First loop gets offsets for AO rays
		for (int i = 0; i < raysIn->GetCount(); i++)
		{
			ao_offsets[i] = nOutputRays;

			if (rays->type[i] == EXTERNAL_RAY_PRIMARY)
			{
				if (rays->term[i] & EXTERNAL_RAY_SURFACE)
					nOutputRays += n_ao_rays;
			}
		}

		int ao_ray_knt = nOutputRays;

		if (do_shadows)
		{
			for (int i = 0; i < raysIn->GetCount(); i++)
			{
				if ((rays->type[i] == EXTERNAL_RAY_PRIMARY) && (rays->term[i] & EXTERNAL_RAY_SURFACE))
				{
					shadow_offsets[i] = nOutputRays;
					nOutputRays += (do_shadows ? numLights : 0);
				}
				else
					shadow_offsets[i] = -1;
			}
		}

		int shadow_ray_knt = nOutputRays - ao_ray_knt;

		OSPExternalRays raysOut;

	  if (nOutputRays != 0)
		{
			raysOut = ospNewExternalRays();
			raysOut->Allocate(nOutputRays);
		}
		else
			raysOut = NULL;

		if (ao_ray_knt)
			ispc::Renderer_generateAORays(getIE(), raysIn->GetCount(), (struct ispc::ExternalRaySOA *)raysIn, ao_offsets, (struct ispc::ExternalRaySOA *)raysOut);
		else
			ispc::Renderer_ambientLighting(getIE(), raysIn->GetCount(), (struct ispc::ExternalRaySOA *)raysIn);

		if (shadow_ray_knt)
			ispc::Renderer_generateShadowRays(getIE(), raysIn->GetCount(), (struct ispc::ExternalRaySOA *)raysIn, shadow_offsets, (struct ispc::ExternalRaySOA *)raysOut);
		else
			ispc::Renderer_diffuseLighting(getIE(), raysIn->GetCount(), (struct ispc::ExternalRaySOA *)raysIn);

#if 0
		if (raysOut)
			std::cerr << "raysOut->count = " << raysOut->GetCount() << "\n";
		else
			std::cerr << "no rays out\n";
#endif

		return raysOut;
	}

  // A renderer type for rays in / rays out rendering
  OSP_REGISTER_RENDERER(PTracerRenderer, ptracer);

} // ::ospray


