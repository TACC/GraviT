//
//  RayTracerAttributes.C
//


#include "RayTracerAttributes.h"

#define HACK_TRANSFER_FUNC                     \
    transfer_func = new unsigned char[4*256];  \
    for (int i=0; i < 128; ++i)                \
    {                                          \
        transfer_func[4*i+0] = 0;              \
        transfer_func[4*i+1] = 2*i;            \
        transfer_func[4*i+2] = 255 - 2*i;      \
        transfer_func[4*i+3] = i;              \
    }                                          \
    for (int i=0; i < 128; ++i)                \
    {                                          \
        transfer_func[128+4*i+0] = 2*i;        \
        transfer_func[128+4*i+1] = 255 - 2*i;  \
        transfer_func[128+4*i+2] = 0;          \
        transfer_func[128+4*i+3] = 2*i;        \
    }                                          \


GVT::Env::RayTracerAttributes* GVT::Env::RayTracerAttributes::rta = NULL;



namespace GVT {
    namespace Env {

        RayTracerAttributes::RayTracerAttributes() {
            // topology should be defined in config file
            topology[0] = -1;
            topology[1] = -1;
            topology[2] = -1;

            // XXX TODO: hacked transfer function.  Fix it.
            HACK_TRANSFER_FUNC;

            do_lighting = false;
        }

        RayTracerAttributes::RayTracerAttributes(string& datafile_, View& view_, RenderType rt = Volume, ScheduleType st = Image, float rate = 1.f, float ratio = 1.f, float* topo = NULL)
        : view(view_), render_type(rt), schedule(st), sample_rate(rate), sample_ratio(ratio) {
            if(rta != NULL) {
                delete rta;
            }
            rta = this;
            if (topo != NULL) {
                topology[0] = topo[0];
                topology[1] = topo[1];
                topology[2] = topo[2];
            }
            HACK_TRANSFER_FUNC;
            do_lighting = false;
        }

        RayTracerAttributes::~RayTracerAttributes() {
            delete[] transfer_func;
        }

        ostream&
        operator<<(ostream& os, RayTracerAttributes::View const& vi) {
            os << vi.width << " x " << vi.height << ", " << vi.view_angle << " angle\n";
            os << "camera: " << vi.camera[0] << " " << vi.camera[1] << " " << vi.camera[2] << "\n";
            os << " focus: " << vi.focus[0] << " " << vi.focus[1] << " " << vi.focus[2] << "\n";
            os << "    up: " << vi.up[0] << " " << vi.up[1] << " " << vi.up[2] << "\n";

            return os;
        }

        istream&
        operator>>(istream& is, RayTracerAttributes::View& vi) {
            is >> vi.width >> vi.height;
            is >> vi.view_angle;
            is >> vi.camera[0] >> vi.camera[1] >> vi.camera[2];
            is >> vi.focus[0] >> vi.focus[1] >> vi.focus[2];
            is >> vi.up[0] >> vi.up[1] >> vi.up[2];
            return is;
        }

        ostream&
        operator<<(ostream& os, RayTracerAttributes const& rta) {
            os << rta.view;
            os << "render type: ";
            switch (rta.render_type) {
                case RayTracerAttributes::Volume:
                    os << "Volume";
                    break;
                case RayTracerAttributes::Surface:
                    os << "Surface";
                    break;
                case RayTracerAttributes::Manta:
                    os << "Surface";
                    break;
                default:
                    os << "Unknown (" << rta.render_type << ")";
            }
            os << "\n";

            os << "schedule: ";
            switch (rta.schedule) {
                case RayTracerAttributes::Image:
                    os << "Image";
                    break;
                case RayTracerAttributes::Domain:
                    os << "Domain";
                    break;
                case RayTracerAttributes::Greedy:
                    os << "Greedy (deprecated)";
                    break;
                case RayTracerAttributes::Spread:
                    os << "Spread (deprecated)";
                    break;
                case RayTracerAttributes::RayWeightedSpread:
                    os << "RayWeightedSpread (deprecated, use LoadOnce)";
                    break;
                case RayTracerAttributes::AdaptiveSend:
                    os << "AdaptiveSend (deprecated)";
                    break;
                case RayTracerAttributes::LoadOnce:
                    os << "LoadOnce";
                    break;
                case RayTracerAttributes::LoadAnyOnce:
                    os << "LoadAnyOnce";
                    break;
                case RayTracerAttributes::LoadAnother:
                    os << "LoadAnother";
                    break;
                case RayTracerAttributes::LoadMany:
                    os << "LoadMany (beta)";
                    break;
                default:
                    os << "Unknown (" << rta.schedule << ")";
            }
            os << "\n";

            os << "sample rate: " << rta.sample_rate;

            os << "dataset: " << rta.dataset << "\n";

            return os;
        }

        istream&
        operator>>(istream& is, RayTracerAttributes& rta) {
            is >> rta.view;

            string rt;
            is >> rt;
            if (rt.find("Volume") != string::npos) {
                rta.render_type = RayTracerAttributes::Volume;
            } else if (rt.find("Surface") != string::npos) {
                rta.render_type = RayTracerAttributes::Surface;
            } else if (rt.find("Manta") != string::npos) {
                rta.render_type = RayTracerAttributes::Manta;
            } else {
                cerr << "Unknown render type '" << rt << "', defaulting to Volume" << endl;
                rta.render_type = RayTracerAttributes::Volume;
            }

            string sch;
            is >> sch;
            if (sch.find("Image") != string::npos)
                rta.schedule = RayTracerAttributes::Image;
            else if (sch.find("Domain") != string::npos)
                rta.schedule = RayTracerAttributes::Domain;
            else if (sch.find("Greedy") != string::npos)
                rta.schedule = RayTracerAttributes::Greedy;
            else if (sch.find("Spread") != string::npos)
                rta.schedule = RayTracerAttributes::Spread;
            else if (sch.find("RayWeightedSpread") != string::npos)
                rta.schedule = RayTracerAttributes::RayWeightedSpread;
            else if (sch.find("AdaptiveSend") != string::npos)
                rta.schedule = RayTracerAttributes::AdaptiveSend;
            else if (sch.find("LoadOnce") != string::npos)
                rta.schedule = RayTracerAttributes::LoadOnce;
            else if (sch.find("LoadAnyOnce") != string::npos)
                rta.schedule = RayTracerAttributes::LoadAnyOnce;
            else if (sch.find("LoadAnother") != string::npos)
                rta.schedule = RayTracerAttributes::LoadAnother;
            else if (sch.find("LoadMany") != string::npos)
                rta.schedule = RayTracerAttributes::LoadMany;
            else {
                cerr << "Unknown schedule '" << sch << "', defaulting to Image" << endl;
                rta.schedule = RayTracerAttributes::Image;
            }

            is >> rta.sample_rate >> rta.sample_ratio;

            is >> rta.topology[0] >> rta.topology[1] >> rta.topology[2];

            
            is >> rta.datafile;

            return is;
        }
    }
}