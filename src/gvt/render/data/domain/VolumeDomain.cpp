//
// Domain.C
//


#include <gvt/render/data/domain/VolumeDomain.h>

#include <cfloat>
#include <fstream>
#include <iostream>

using namespace gvt::render::data::domain;

bool
X_Box( const gvt::render::actor::Ray& r, const float* min, const float* max, 
       float& t_near, float& t_far )
{
    t_near = -FLT_MAX;
    t_far = FLT_MAX;
    
    for (int i=0; i < 3; ++i)
    {
        if ( r.direction[i] == 0)
        {
            if ((r.origin[i] < min[i]) | (r.origin[i] > max[i]))
                return false;
        }
        else
        {
            float inv_d = 1.f / r.direction[i];
            float t1, t2;
            t1 = (min[i] - r.origin[i]) * inv_d;
            t2 = (max[i] - r.origin[i]) * inv_d;
            if (t1 > t2)
            {
                float temp = t1;
                t1 = t2;
                t2 = temp;
            }

            if (t1 > t_near) t_near = t1;
            if (t2 < t_far)  t_far = t2;
            if (t_near > t_far) return false;
            if (t_far < 0) return false;
        }       
    }

    return true;
}

bool
VolumeDomain::MakeCell(int id, Cell& cell) 
{
    return Cell::MakeCell(id, cell, dim, min, max, cell_dim, data);
}

bool
VolumeDomain::intersect(gvt::render::actor::Ray& r, std::vector<int>& cells) 
{
    float near, far;
    float p_min[3] = {this->min[0], this->min[1], this->min[2]};
    float p_max[3] = {this->max[0], this->max[1], this->max[2]};

    if (!X_Box(r, p_min, p_max, near, far)) 
    {
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

    for (int i = 0; i < 3; ++i) 
    {
        if (r.direction[i] == 0) 
        {
            tMax[i] = FLT_MAX;
            tDelta[i] = 0;
        }
        if (r.direction[i] < 0) 
        {
            step[i] *= -1;
            justOut[i] = -1;
        }
    }

    do 
    {
        int cell_idx = idx_near[0] + idx_near[1] * dim[0] + idx_near[2] * dim[0] * dim[1];
        cells.push_back(cell_idx);

        if (tMax[0] < tMax[1]) 
        {
            if (tMax[0] < tMax[2]) 
            {
                idx_near[0] += step[0];
                tMax[0] += tDelta[0];
            } 
            else 
            {
                idx_near[2] += step[2];
                tMax[2] += tDelta[2];
            }
        } 
        else 
        {
            if (tMax[1] < tMax[2]) 
            {
                idx_near[1] += step[1];
                tMax[1] += tDelta[1];
            }
            else 
            {
                idx_near[2] += step[2];
                tMax[2] += tDelta[2];
            }
        }

    } 
    while (idx_near[0] != justOut[0]
            && idx_near[1] != justOut[1]
            && idx_near[2] != justOut[2]);

    return true;
}

// XXX TODO: currently assumes only brick of floats

bool
VolumeDomain::load() 
{
    // data already loaded
    if (data != NULL) return true;

    std::streampos len;
    std::ifstream in;
    in.open(filename.c_str(), std::ios::binary);

    if (!in.good()) 
    {
        GVT_DEBUG(DBG_ALWAYS,"ERROR: failed to open domain file '" << filename << "'");
        return false;
    }

    in.seekg(0, std::ios::end);
    len = in.tellg();
    in.seekg(0, std::ios::beg);

    if ((int) len != this->sizeInBytes()) 
    {
        GVT_DEBUG(DBG_ALWAYS,"ERROR: File size mismatch!"
            << "  Expected " << (dim[0] * dim[1] * dim[2] * sizeof (float))
            << "  but got " << len);
        return false;
    }

    data = new float[len];
    in.read((char*) data, len);

    in.close();
    return true;
}

namespace gvt{ namespace render{ namespace data{ namespace domain{
std::ostream&
operator<<(std::ostream& os, VolumeDomain const& d) 
{
    os << "volume domain " << d.id << std::endl;
    os << "    file: " << d.filename << std::endl;
    os << "    dim: " << d.dim[0] << " " << d.dim[1] << " " << d.dim[2] << std::endl;
    os << "    min: " << d.min[0] << " " << d.min[1] << " " << d.min[2] << std::endl;
    os << "    max: " << d.max[0] << " " << d.max[1] << " " << d.max[2] << std::endl;
    os << "    cell_dim: " << d.cell_dim[0] << " " << d.cell_dim[1] << " " << d.cell_dim[2] << std::endl;
    os << std::flush;

    return os;
}
}}}} // namespace domain} namespace data} namespace render} namespace gvt}