import gvt
import reader as rd
import sys
import numpy as np
import argparse

# parse command line
parser = argparse.ArgumentParser(description='A PYGVT example to render a model in obj/ply/vtp format.')
parser.add_argument('filename', help='file name')
parser.add_argument('--image-size', nargs=2, default=[512, 512], help='image width and height')
parser.add_argument('--light-color', nargs=3, default=[100.0, 100.0, 100.0], help='light color')
parser.add_argument('--diffuse-color', nargs=3, default=[1.0, 1.0, 1.0], help='diffuse color for surfaces')
parser.add_argument('--camera-distance', nargs=1, default=1.0, help='camera distance factor (the larger the farther)')
args = parser.parse_args()

# set filename
filename = args.filename

# read file
r = rd.Reader()
r.read(filename)
r.print_stats()

# initialize gvt
gvt.gvtInit()

# diffuse color
kd = np.array(args.diffuse_color, dtype=np.float32);

# create a mesh
mesh_name = r.mesh_name
gvt.createMesh(mesh_name)

# add vertices and triangles
gvt.addMeshVertices(mesh_name, len(r.vertices) / 3, r.vertices)
gvt.addMeshTriangles(mesh_name, len(r.triangles) / 3, r.triangles)

# set material
gvt.addMeshMaterialLambert(mesh_name, 0, kd, 1.0)
gvt.finishMesh(mesh_name)

# transformation matrix
gvt.addInstance(mesh_name, np.array([1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0], dtype=np.float32))

# camera
camera_pos = (r.center + (np.array([0.0, 0.0, 1.0], dtype=np.float32) * (args.camera_distance * r.diagonal))).astype(np.float32)
camera_focus = r.center.astype(np.float32)

gvt.addCamera("Camera", camera_pos, camera_focus,
               # up
               np.array([0.0, 1.0, 0.0], dtype=np.float32),
               # fov, depth, samples, jitter
               0.785398, 1, 1, 0.5)

# light source
# TODO: make this configurable
# locate a point light at the camera position
# light_pos = (r.center + (np.array([0.0, 1.0, 0.0], dtype=np.float32) * r.diagonal)).astype(np.float32)
light_pos = camera_pos
# TODO: make this configurable
light_color = np.array(args.light_color, dtype=np.float32)

gvt.addPointLight("PointLight", light_pos, light_color)

# set image size
gvt.addFilm("film", args.image_size[0], args.image_size[1], mesh_name)

# add renderer
gvt.addRenderer("render", 4, 0) # name, adapter, schedule;
gvt.render("render");

# dump the image to a ppm file
gvt.writeimage("render", mesh_name);

print("created " + mesh_name + ".ppm")


