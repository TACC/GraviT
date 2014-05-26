//
// GeometryDomain.h
//

#ifndef GVT_GEOMETRY_DOMAIN_H
#define GVT_GEOMETRY_DOMAIN_H

#include "Domain.h"
#include <GVT/Data/primitives.h>
#include <GVT/utils/readply.h>
#include <vector>
namespace GVT {
    namespace Domain {

        class GeometryDomain : public Domain {
        public:

            GVT::Data::Mesh* mesh;
            std::vector<GVT::Data::lightsource*> lights;
            GVT::Data::box3D boundingBox;
            std::string filename;
            bool domain_loaded;

            GeometryDomain(std::string filename = "", GVT::Math::AffineTransformMatrix<float> m = GVT::Math::AffineTransformMatrix<float>(true)) : Domain(m), mesh(NULL), filename(filename), domain_loaded(false) {
                if (filename != "") {
                    LoadData();
                    FreeData();
                }
            }

            virtual ~GeometryDomain() {

            }

            GeometryDomain(const GeometryDomain& other) : Domain(other) {
                mesh = other.mesh;
                lights = other.lights;
                boundingBox = other.boundingBox;
                filename = other.filename;
                domain_loaded = other.domain_loaded;
            }

            virtual bool Intersect(GVT::Data::ray&, vector<int>&);

            virtual bool hasGeometry() {
                return domain_loaded;
            }

            virtual int Size() {
                return 0;
            }

            virtual int SizeInBytes() {
                return 0;
            }

            virtual bool LoadData() {
                if (domain_loaded) return true;
                if (filename == "") return false;
                mesh = readply(filename);
                lights.push_back(new GVT::Data::PointLightSource(GVT::Math::Point4f(5.0, 5.0, 5.0, 1.f), GVT::Data::Color(1.f, 1.f, 1.f, 1.f)));
                mesh->mat = new GVT::Data::Lambert(GVT::Data::Color(1.f, .0f, .0f, 1.f));
                boundingBox = mesh->boundingBox;
                domain_loaded = true;
                return true;
            }

            virtual void FreeData();

            GVT::Data::box3D getBounds(int type = 0) {
                if (type == 0) {
                    return boundingBox;
                } else {
                    GVT::Data::box3D bb; // = boundingBox;
                    bb.bounds[0] = m * boundingBox.bounds[0];
                    bb.bounds[1] = m * boundingBox.bounds[1];
                    return bb;
                    
                }
            }

            friend ostream& operator<<(ostream&, GeometryDomain const&);

            virtual void operator()() {
            }

        };
    };
};

#endif // GVT_GEOMETRY_DOMAIN_H
