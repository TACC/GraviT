/* 
 * File:   OptixDataset.h
 * Author: jbarbosa
 *
 * Created on September 20, 2014, 5:36 PM
 */

#ifndef OPTIXDATASET_H
#define	OPTIXDATASET_H

#include <GVT/DataSet/Dataset.h>

namespace GVT {
    namespace Dataset {

        class OptixDataset : public GVTDataset {
        public:

            OptixDataset() {
            }

            OptixDataset(string& filename) : GVTDataset(), conf_filename(filename) {
                GVT_DEBUG(DBG_ALWAYS, "Filename : " + filename);
                conf_filename = filename;
            }

            virtual bool init();

        private:
            vector<string> files;
            string conf_filename;
        };
    }
}

#endif	/* OPTIXDATASET_H */

