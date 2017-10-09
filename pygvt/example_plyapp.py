import gvt
import reader as rd
import sys
import os
import numpy as np
import argparse

# parse command line
parser = argparse.ArgumentParser(description='A PYGVT example to render a model in obj/ply/vtp format.')
parser.add_argument('dirname', help='directory name')
parser.add_argument('--image-size', nargs=2, default=[512, 512], help='image width and height')
parser.add_argument('--light-color', nargs=3, default=[100.0, 100.0, 100.0], help='light color')
parser.add_argument('--diffuse-color', nargs=3, default=[1.0, 1.0, 1.0], help='diffuse color for surfaces')
parser.add_argument('--camera-distance', nargs=1, type=float, default=[1.0], help='camera distance factor (the larger the farther)')
args = parser.parse_args()

# set filename
# filename = args.filename
dirname = args.dirname

# initialize gvt
gvt.gvtInit()
 
# create a mesh
model_name = os.path.basename(dirname).split(".")[0]
 
bounds = np.zeros((6,), dtype=np.float32)
gvt.readPly(dirname, False, bounds)

bounds = np.reshape(bounds, (2,3))
diagonal = np.linalg.norm(bounds[1] - bounds[0])
center = (bounds[0] + bounds[1]) * 0.5

print("bounds_min: ", bounds[0])
print("bounds_max: ", bounds[1])
print("diagonal: ", diagonal)
print("center: ", center)

# camera
camera_pos = (center + (np.array([0.0, 0.0, 1.0], dtype=np.float32) * (args.camera_distance[0] * diagonal))).astype(np.float32)
camera_focus = center.astype(np.float32)

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
gvt.addFilm("film", args.image_size[0], args.image_size[1], model_name)

# add renderer
gvt.addRenderer("render", 4, 0) # name, adapter, schedule;
gvt.render("render");

# dump the image to a ppm file
gvt.writeimage("render", model_name);

print("created " + model_name + ".ppm")


