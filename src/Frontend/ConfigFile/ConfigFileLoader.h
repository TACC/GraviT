/* 
 * File:   ConfigFileLoader.h
 * Author: jbarbosa
 *
 * Created on January 21, 2015, 12:15 PM
 */

#ifndef CONFIGFILELOADER_H
#define	CONFIGFILELOADER_H

#include <string>

#include <GVT/DataSet/Dataset.h>
#include <GVT/Domain/domains.h>
#include <GVT/common/debug.h>

namespace GVT {
    namespace Frontend {

        class ConfigFileLoader {
        public:
            ConfigFileLoader(const std::string filename = "");
            ConfigFileLoader(const ConfigFileLoader& orig);
            virtual ~ConfigFileLoader();
        //private:

            GVT::Dataset::GVTDataset scene;
            //GVT::Env::RayTracerAttributes rta;

        };

    };
};

#endif	/* CONFIGFILELOADER_H */

