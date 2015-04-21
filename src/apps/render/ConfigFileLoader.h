/* 
 * File:   ConfigFileLoader.h
 * Author: jbarbosa
 *
 * Created on January 21, 2015, 12:15 PM
 */

#ifndef GVTAPPS_RENDER_CONFIG_FILE_LOADER_H
#define	GVTAPPS_RENDER_CONFIG_FILE_LOADER_H

#include <string>

#include <gvt/render/data/Dataset.h>
#include <gvt/render/data/Domains.h>
#include <gvt/core/Debug.h>

namespace gvtapps {
    namespace render {

        class ConfigFileLoader {
        public:
            ConfigFileLoader(const std::string filename = "");
            ConfigFileLoader(const ConfigFileLoader& orig);
            virtual ~ConfigFileLoader();
        //private:

            gvt::render::data::Dataset scene;
            //GVT::Env::RayTracerAttributes rta;

        };

    }
}

#endif	/* GVTAPPS_RENDER_CONFIG_FILE_LOADER_H */

