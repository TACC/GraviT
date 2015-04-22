/* 
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

/// GVT configuration file loader
/** Load geometric data from a gvt configuration file. The configuration file contains "scene" information such as camera descriptions, lights, 
 * and descriptions of geometric objects in the scene. The components are loaded into a render dataset object. The config file that is read 
 * by this class also contains information on which renderer to use. Back end renderers such as Optix and Manta do the actual ray-geometry 
 * intersections. The configuration file loader also adapts to object types and uses the appropriate reader class to parse the geometric
 * input. The extension of the file name in the config file is used to select which geometry reader to use. For example an .obj file extension
 * would cause the gvt obj file reader to be used to parse that file.  
 * File:   ConfigFileLoader.h
*/
        class ConfigFileLoader {
        public:
	    /** Constructor that utilizes the file name of the gvt config file.
	    */
            ConfigFileLoader(const std::string filename = "");
	    /** Copy constructor. 
	    */
            ConfigFileLoader(const ConfigFileLoader& orig);
            virtual ~ConfigFileLoader();
        //private:
	    /** gvt render dataset member function that contains all the scene data, cameras, lights, objects etc.
	    */
            gvt::render::data::Dataset scene;
            //GVT::Env::RayTracerAttributes rta;
            
	    /** Private data member that indicates the type of renderer to use. Options include but are not limited to
	    * Manta, Optix, and Embree.
	    */
            unsigned domain_type = 0 ; /* default Manta domain */
	    /** Private data member that indicates the type of scheduler to use. The default scheduler is the Image scheduler
	    */
            unsigned scheduler_type = 0 ; /* default Image scheduler */


        };

    }
}

#endif	/* GVTAPPS_RENDER_CONFIG_FILE_LOADER_H */

