import gvt
import numpy as np


gvt.gvtInit()


vertices = np.array([
    0.5,     0.0,  0.0,
    -0.5, 0.5,  0.0,
    -0.5,      0.25, 0.433013,
    -0.5, -0.25, 0.43013,
    -0.5, -0.5, 0.0,
    -0.5, -0.25, -0.433013,
    -0.5, 0.25, -0.433013 ],
    dtype=np.float32)

triangles = np.array([1, 2, 3, 1, 3, 4, 1, 4, 5, 1, 5, 6, 1, 6, 7, 1, 7, 2 ], dtype=np.uint32)
kd = np.array([ 1.0, 1.0, 1.0 ], dtype=np.float32);





vertices_cube = np.array([ -0.5, -0.5, 0.5,  0.5,  -0.5, 0.5,  0.5,  0.5,  0.5,  -0.5, 0.5,  0.5,
-0.5, -0.5, -0.5, 0.5,  -0.5, -0.5, 0.5,  0.5,  -0.5, -0.5, 0.5,  -0.5,
0.5,  0.5,  0.5,  -0.5, 0.5,  0.5,  0.5,  0.5,  -0.5, -0.5, 0.5,  -0.5,
-0.5, -0.5, 0.5,  0.5,  -0.5, 0.5,  -0.5, -0.5, -0.5, 0.5,  -0.5, -0.5,
0.5,  -0.5, 0.5,  0.5,  0.5,  0.5,  0.5,  -0.5, -0.5, 0.5,  0.5,  -0.5,
-0.5, -0.5, 0.5,  -0.5, 0.5,  0.5,  -0.5, -0.5, -0.5, -0.5, 0.5,  -0.5
],dtype=np.float32)

triangles_cube = np.array([
1,  2,  3,  1,  3,  4,  17, 19, 20, 17, 20, 18, 6,  5,  8,  6,  8,  7,
23, 21, 22, 23, 22, 24, 10, 9,  11, 10, 11, 12, 13, 15, 16, 13, 16, 14,
],dtype=np.uint32);

gvt.createMesh("ConeMesh")
gvt.addMeshVertices("ConeMesh",len(vertices)/3,vertices)
gvt.addMeshTriangles("ConeMesh",len(triangles)/3,triangles)
gvt.addMeshMaterialLambert("ConeMesh",0,kd,1.0)
gvt.finishMesh("ConeMesh")

gvt.createMesh("CubeMesh")
gvt.addMeshVertices("CubeMesh",len(vertices_cube)/3,vertices_cube)
gvt.addMeshTriangles("CubeMesh",len(triangles_cube)/3,triangles_cube)
gvt.addMeshMaterialLambert("CubeMesh",0,kd,1.0)
gvt.finishMesh("CubeMesh")

gvt.addInstance("Cone1", "ConeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, -1, -1, 1],dtype=np.float32))
gvt.addInstance("Cube1", "CubeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, -1, -0.5, 1],dtype=np.float32))
gvt.addInstance("Cone2", "ConeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, -1, 0, 1],dtype=np.float32))
gvt.addInstance("Cube3", "CubeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, -1, 0.5, 1],dtype=np.float32))
gvt.addInstance("Cone3", "ConeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, -1, 1, 1],dtype=np.float32))
gvt.addInstance("Cube4", "CubeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, -0.5, -1, 1],dtype=np.float32))
gvt.addInstance("Cone4", "ConeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, -0.5, -0.5, 1],dtype=np.float32))
gvt.addInstance("Cube5", "CubeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, -0.5, 0, 1],dtype=np.float32))
gvt.addInstance("Cone5", "ConeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, -0.5, 0.5, 1],dtype=np.float32))
gvt.addInstance("Cone6", "ConeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, -1, 1],dtype=np.float32))
gvt.addInstance("Cube6", "CubeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, -0.5, 1],dtype=np.float32))
gvt.addInstance("Cone7", "ConeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 1],dtype=np.float32))
gvt.addInstance("Cube7", "CubeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0.5, 1],dtype=np.float32))
gvt.addInstance("Cone8", "ConeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 1, 1],dtype=np.float32))
gvt.addInstance("Cube8", "CubeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, 0.5, -1, 1],dtype=np.float32))
gvt.addInstance("Cone9", "ConeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, 0.5, -0.5, 1],dtype=np.float32))
gvt.addInstance("Cube9", "CubeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, 0.5, 0, 1],dtype=np.float32))
gvt.addInstance("Cone10", "ConeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, 0.5, 0.5, 1],dtype=np.float32))
gvt.addInstance("Cube10", "CubeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, 0.5, 1, 1],dtype=np.float32))
gvt.addInstance("Cone11", "ConeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, 1, -1, 1],dtype=np.float32))
gvt.addInstance("Cone12", "ConeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, 1, 0, 1],dtype=np.float32))
gvt.addInstance("Cube11", "CubeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, 1, 0.5, 1],dtype=np.float32))
gvt.addInstance("Cone13", "ConeMesh", np.array([0.4, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0.4, 0, 0, 1, 1, 1],dtype=np.float32))

gvt.addPointLight("LightCone",np.array([1.0, 0.0, -1.0],dtype=np.float32),np.array([1.0, 1.0, 1.0],dtype=np.float32))

gvt.addCamera("MyCamera",np.array([4.0, 0.0, 0.0],dtype=np.float32), np.array([0.0, 0.0, 0.0],dtype=np.float32), np.array([0.0, 1.0, 0.0],dtype=np.float32), 0.785398, 1, 1,
            0.5)

gvt.addFilm("film",512,512,"simple")
gvt.addRenderer("render", 4, 0);
gvt.render("render");
gvt.writeimage("render","simple");
