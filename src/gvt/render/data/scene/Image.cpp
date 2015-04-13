//
// Image.C
//

#include <gvt/render/data/scene/Image.h>

#include <gvt/core/Debug.h>

#include <fstream>
#include <iostream>
#include <sstream>

using namespace gvt::render::data::scene;

void
Image::Write()
{
    std::string ext;
    switch (format)
    {
    case PPM:
        ext = ".ppm"; break;
    default:
        GVT_DEBUG(DBG_ALWAYS,"ERROR: unknown image format '" << format << "'");
        return;
    }

    std::stringstream header;
    header << "P6" << std::endl;
    header << width << " " << height << std::endl;
    header << "255" << std::endl;

    std::fstream file;
    file.open( (filename + ext).c_str(), std::fstream::out | std::fstream::trunc | std::fstream::binary );
    file << header.str();
    
    // reverse row order so image is correctly oriented
    for (int j=height-1; j >= 0; --j)
    {
        int offset = j*width;
        for (int i=0; i < width; ++i)
        {
            int index = 3*(offset+i);
            file << rgb[index+0] << rgb[index+1] << rgb[index+2];
        }
    }

    file.close();
}
