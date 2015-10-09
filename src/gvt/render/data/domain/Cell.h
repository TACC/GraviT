/* 
 * File:   cell.h
 * Author: jbarbosa
 *
 * Created on February 27, 2014, 12:06 PM
 */

#ifndef GVT_RENDER_DATA_DOMAIN_CELL_H
#define	GVT_RENDER_DATA_DOMAIN_CELL_H


#include <gvt/render/data/Primitives.h>

#include <iostream>
#include <vector>

namespace gvt {
    namespace render {
        namespace data {
            namespace domain {
                /// subdivision of a domain
                class Cell 
                {
                public:

                    float data[8];
                    float min[3];
                    float max[3];

                    Cell() {}
                    Cell(int id, Cell& cell, float *) {}
                    Cell(const Cell& other);
                    virtual ~Cell();

                    class Face 
                    {
                    public:

                        Face(float t_ = 0, float p1 = 0, float p2 = 0, float d1 = 0, float d2 = 0, float d3 = 0, float d4 = 0) 
                        {
                            t = t_;
                            pt[0] = p1;
                            pt[1] = p2;
                            data[0] = d1;
                            data[1] = d2;
                            data[2] = d3;
                            data[3] = d4;
                        }

                        Face(const Face& f) 
                        {
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

                    bool FindFaceIntersectionsWithRay(gvt::render::actor::Ray&, std::vector<Face>&);

                    static bool MakeCell(int id, Cell& cell, std::vector<int>& dim, std::vector<float> min, std::vector<float> max, std::vector<float>& cell_dim, float* data);

                    friend std::ostream& operator<<(std::ostream&, Face const&);
                };
            }
        }
    }
}

#endif	/* GVT_RENDER_DATA_DOMAIN_CELL_H */

