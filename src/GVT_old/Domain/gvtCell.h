/* 
 * File:   cell.h
 * Author: jbarbosa
 *
 * Created on February 27, 2014, 12:06 PM
 */

#ifndef GVT_CELL_H
#define	GVT_CELL_H


#include <GVT/Data/primitives.h>
#include <GVT/Data/scene/Utils.h>

#include <iostream>
#include <vector>

namespace GVT {
    namespace Domain {

        class gvtCell {
        public:

            float data[8];
            float min[3];
            float max[3];

            gvtCell() {
            };

            gvtCell(int id, gvtCell& cell, float *) {
            };
            gvtCell(const gvtCell& other);
            virtual ~gvtCell();

            class Face {
            public:

                Face(float t_ = 0, float p1 = 0, float p2 = 0, float d1 = 0, float d2 = 0, float d3 = 0, float d4 = 0) {
                    t = t_;
                    pt[0] = p1;
                    pt[1] = p2;
                    data[0] = d1;
                    data[1] = d2;
                    data[2] = d3;
                    data[3] = d4;
                }

                Face(const Face& f) {
                    t = f.t;
                    pt[0] = f.pt[0];
                    pt[1] = f.pt[1];
                    data[0] = f.data[0];
                    data[1] = f.data[1];
                    data[2] = f.data[2];
                    data[3] = f.data[3];
                }

                float t;
                float pt[2];
                float data[4];
            };

            bool FindFaceIntersectionsWithRay(GVT::Data::ray&, vector<Face>&);

            static bool MakeCell(int id, gvtCell& cell, vector<int>& dim, vector<float> min, vector<float> max, vector<float>& cell_dim, float* data);




            friend ostream& operator<<(ostream&, Face const&);
        };

    }
}
#endif	/* GVT_CELL_H */

