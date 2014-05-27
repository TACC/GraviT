//
//  RayTracerAttributes.h
//

#ifndef GVT_RAY_TRACER_ATTRIBUTES_H
#define GVT_RAY_TRACER_ATTRIBUTES_H

#include <GVT/DataSet/Dataset.h>
#include <GVT/Data/primitives.h>

#include <GVT/Math/GVTMath.h>

#include <iostream>
using namespace std;
namespace GVT {
    namespace Env {

        class RayTracerAttributes {
        public:

            class View {
            public:

                View() {
                }

                View(const View& v) {
                    width = v.width;
                    height = v.height;
                    view_angle = v.view_angle;
                    camera[0] = v.camera[0];
                    camera[1] = v.camera[1];
                    camera[2] = v.camera[2];
                    focus[0] = v.focus[0];
                    focus[1] = v.focus[1];
                    focus[2] = v.focus[2];
                    up[0] = v.up[0];
                    up[1] = v.up[1];
                    up[2] = v.up[2];
                }

                friend ostream& operator<<(ostream&, View const&);
                friend istream& operator>>(istream&, View&);

                int width, height;
                int view_angle;
                GVT::Math::Point4f camera;
                GVT::Math::Point4f focus;
                GVT::Math::Vector4f up;

                float parallelScale;
                float nearPlane;
                float farPlane;
            };

            enum RenderType {
                Volume,
                Surface,
                Manta
            };

            enum ScheduleType {
                Image,
                Domain,
                Greedy, // PAN: deprecated
                Spread, // PAN: deprecated
                RayWeightedSpread, // PAN: from EGPGV 2012 paper, deprecated, now called LoadOnce
                AdaptiveSend, // PAN: deprecated
                LoadOnce, // PAN: from TVCG 2013 paper
                LoadAnyOnce, // PAN: from TVCG 2013 paper
                LoadAnother, // PAN: from TVCG 2013 paper
                LoadMany,
            };

            RayTracerAttributes();
            RayTracerAttributes(string&, View&, RenderType, ScheduleType, float, float, float*);
            ~RayTracerAttributes();

            bool LoadDataset() {
                
                GVT_DEBUG(DBG_ALWAYS,"Sent load");
                
                return dataset->init();
            }

            ScheduleType GetSchedule() {
                return schedule;
            }

            unsigned char* GetTransferFunction() {
                return transfer_func;
            }

            int* GetTopology() {
                return topology;
            }

            void GetVarExtents(float& vmin, float& vmax) {
                vmin = 1.09554;
                vmax = 5.88965;
            } // XXX TODO: hacked for noise.conf

            void GetOpacityExtents(float& omin, float& omax) {
                omin = 1.09554;
                omax = 5.88965;
            } // XXX TODO: hacked for noise.conf

            void SetLightingFlag(bool doit) {
                do_lighting = doit;
            }

            bool GetLightingFlag() {
                return do_lighting;
            }

            float GetReflectivityThreshold() {
                return 1.;
            }

            friend ostream& operator<<(ostream&, RayTracerAttributes const&);
            friend istream& operator>>(istream&, RayTracerAttributes&);
            friend class RayTracer;

        public:
            View view;
            RenderType render_type;
            ScheduleType schedule;
            float sample_rate;
            float sample_ratio;
            GVT::Dataset::GVTDataset* dataset;
            std::string datafile;
            int topology[3];
            unsigned char* transfer_func;

            bool do_lighting;

        };
    };
};
#endif // GVT_RAY_TRACER_ATTRIBUTES_H
