//
//  Color.C
//

#include <gvt/render/data/scene/ColorAccumulator.h>

using namespace gvt::render::data::scene;
namespace gvt{ namespace render{ namespace data{ namespace scene{
const float ColorAccumulator::ALPHA_MAX = 0.999;

std::ostream& operator<<(std::ostream &out, ColorAccumulator const& c) 
{
    out << "(t:" << c.t << ") r:" << c.rgba[0] << " g:" << c.rgba[1] << " b:" << c.rgba[2] << " a:" << c.rgba[3];
    return out;
}
}}}} // namespace scene} namespace data} namespace render} namespace gvt}