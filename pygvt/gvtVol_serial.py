#
# gvtVol_serial.py
#
# This application tests the python interface for the volume rendering
# parts of the GraviT API.
#
# run with this command "python gvtVol_serial.py"
#

import gvt
import numpy as np

# A python implementation of the GraviT volume renderer

gvt.gvtInit()

# Read some data

dims = (256, 256, 256)
Volume = np.fromfile("../data/vol/1000.int256", dtype=np.int32).astype(
    np.float32)

# Some volume metadata

# Deltas - cell spacing

deltas = np.array([1.0, 1.0, 1.0], dtype=np.float32)

# Counts - number of points in each coordinate direction

counts = np.array(dims, dtype=np.int32)

# Origin - origin of the volume in world space

origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)

# samplingrate - rate of the sampling

samplingrate = 1.0

# color transfer function

ctf = "../data/colormaps/Grayscale.orig.cmap"

# opacity transfer function

otf = "../data/colormaps/Grayscale.orig.cmap"

# name of the volume node on each rank

volnodename = "1000.int256"

# min and max values of the data.

low = 0.0
high = 65536.0

# make da volume

gvt.createVolume(volnodename)
gvt.addVolumeTransferFunctions(volnodename, ctf, otf, low, high)
gvt.addVolumeSamples(volnodename, Volume, counts, origin, deltas, samplingrate)

# add an instance.
# An instance needs a transformation matrix. All coords of the volume data
# are in world coords so we give the identity matrix.

mf = np.identity(4, dtype=np.float32).flatten()
myinstance = "inst"
gvt.addInstance(myinstance, volnodename, mf)

# Set up camera and film.

eyept = np.array([512.0, 512.0, 512.0], dtype=np.float32)
focus = np.array([127.5, 127.5, 127.5], dtype=np.float32)
fov = 30.0*np.pi/180.0
upVector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
rayMaxDepth = 1
raySamples = 1
jitterWindowSize = 0.5
camname = "volCam"

gvt.addCamera(camname, eyept, focus, upVector, fov, rayMaxDepth, raySamples,
              jitterWindowSize)

# set up film.

wsize = np.array([512, 512], dtype=np.int32)
filmname = "volFilm"
imagename = "PythonVolImage"

gvt.addFilm(filmname, wsize[0], wsize[1], imagename)

# renderer bits ...

rendername = "PythonVolRenderer_serial"

# these are the integer values of the two "types" they are
# enums in the C++. Hardwire here for domain schedul and ospray adapter
schedtype = 1
adaptertype = 5

gvt.addRenderer(rendername, adaptertype, schedtype, camname, filmname, True)

# render it

gvt.render(rendername)

# save the image

gvt.writeimage(rendername, imagename)
