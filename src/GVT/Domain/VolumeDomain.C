//
// Domain.C
//


#include "VolumeDomain.h"

#include <GVT/Data/scene/Utils.h>

#include <cfloat>
#include <fstream>
#include <iostream>


namespace GVT {
    namespace Domain {

        bool
        VolumeDomain::MakeCell(int id, gvtCell& cell) {
            return gvtCell::MakeCell(id, cell, dim, min, max, cell_dim, data);
        }

        bool
        VolumeDomain::Intersect(GVT::Data::ray& r, vector<int>& cells) {
            float near, far;
            float p_min[3] = {this->min[0], this->min[1], this->min[2]};
            float p_max[3] = {this->max[0], this->max[1], this->max[2]};

            if (!X_Box(r, p_min, p_max, near, far)) {
                cells.clear();
                return false;
            }
            GVT_DEBUG(DBG_LOW,"  hit! " << near << "  " << far);

            float p_near[3];
            p_near[0] = r.origin[0] + r.direction[0] * near;
            p_near[1] = r.origin[1] + r.direction[1] * near;
            p_near[2] = r.origin[2] + r.direction[2] * near;

#define ARB_E 1e-3f
            // XXX TODO: cell_dim assumes uniform grid
            int idx_near[3];
            float fudge[3] = {r.direction[0] * ARB_E, r.direction[1] * ARB_E, r.direction[2] * ARB_E}; // put hitpoint just inside cube
            idx_near[0] = (p_near[0] + fudge[0] - p_min[0]) / cell_dim[0];
            idx_near[1] = (p_near[1] + fudge[1] - p_min[1]) / cell_dim[1];
            idx_near[2] = (p_near[2] + fudge[2] - p_min[2]) / cell_dim[2];

            // grid traversal using [Amanatides and Woo 1987]
            int step[3] = {1, 1, 1};
            int justOut[3] = {dim[0], dim[1], dim[2]};
            float tMax[3];
            tMax[0] = ((idx_near[0] + step[0]) * cell_dim[0] - p_near[0]) / r.direction[0];
            tMax[1] = ((idx_near[1] + step[1]) * cell_dim[1] - p_near[1]) / r.direction[1];
            tMax[2] = ((idx_near[2] + step[2]) * cell_dim[2] - p_near[2]) / r.direction[2];
            float tDelta[3];
            tDelta[0] = cell_dim[0] / r.direction[0];
            tDelta[1] = cell_dim[1] / r.direction[1];
            tDelta[2] = cell_dim[2] / r.direction[2];

            for (int i = 0; i < 3; ++i) {
                if (r.direction[i] == 0) {
                    tMax[i] = FLT_MAX;
                    tDelta[i] = 0;
                }
                if (r.direction[i] < 0) {
                    step[i] *= -1;
                    justOut[i] = -1;
                }
            }

            do {
                int cell_idx = idx_near[0] + idx_near[1] * dim[0] + idx_near[2] * dim[0] * dim[1];
                cells.push_back(cell_idx);

                if (tMax[0] < tMax[1]) {
                    if (tMax[0] < tMax[2]) {
                        idx_near[0] += step[0];
                        tMax[0] += tDelta[0];
                    } else {
                        idx_near[2] += step[2];
                        tMax[2] += tDelta[2];
                    }
                } else {
                    if (tMax[1] < tMax[2]) {
                        idx_near[1] += step[1];
                        tMax[1] += tDelta[1];
                    } else {
                        idx_near[2] += step[2];
                        tMax[2] += tDelta[2];
                    }
                }

            } while (idx_near[0] != justOut[0]
                    && idx_near[1] != justOut[1]
                    && idx_near[2] != justOut[2]);

            return true;
        }

        // XXX TODO: currently assumes only brick of floats

        bool
        VolumeDomain::LoadData() {
            // data already loaded
            if (data != NULL) return true;

            streampos len;
            ifstream in;
            in.open(filename.c_str(), ios::binary);

            if (!in.good()) {
                cerr << "ERROR: failed to open domain file '" << filename << "'" << endl;
                return false;
            }

            in.seekg(0, ios::end);
            len = in.tellg();
            in.seekg(0, ios::beg);

            if ((int) len != this->SizeInBytes()) {
                cerr << "ERROR: File size mismatch!";
                cerr << "  Expected " << (dim[0] * dim[1] * dim[2] * sizeof (float));
                cerr << "  but got " << len << endl;
                return false;
            }

            data = new float[len];
            in.read((char*) data, len);

            in.close();
            return true;
        }

        ostream&
        operator<<(ostream& os, VolumeDomain const& d) {
            os << "volume domain " << d.id << endl;
            os << "    file: " << d.filename << endl;
            os << "    dim: " << d.dim[0] << " " << d.dim[1] << " " << d.dim[2] << endl;
            os << "    min: " << d.min[0] << " " << d.min[1] << " " << d.min[2] << endl;
            os << "    max: " << d.max[0] << " " << d.max[1] << " " << d.max[2] << endl;
            os << "    cell_dim: " << d.cell_dim[0] << " " << d.cell_dim[1] << " " << d.cell_dim[2] << endl;
            os << flush;

            return os;
        }
    };
};