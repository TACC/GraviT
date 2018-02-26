#
# gvtVol_yt.py
#
# This application tests the python interface for the volume rendering
# parts of the GraviT API and loads data to render using yt.
#
# Currently it uses the Enzo_64 test dataset and creates a 256^3 uniform
# resolution data cube before passing the data to GraviT to render.
#
# It should be possible to adapt this to work with arbitrary data loadable by
# yt, so long as yt can convert it to a covering grid.

import gvt
import numpy as np
import yt

# A python implementation of the GraviT volume renderer

gvt.gvtInit()

# Read some data

dims = (256, 256, 256)

# sample the full domain (Enzo's internal units scale from 0 to 1 along
# the edge of the domain)
left_corner = (0, 0, 0)
right_corner = (1, 1, 1)
ds = yt.load('Enzo_64/DD0032/data0032')
grid = ds.arbitrary_grid(left_corner, right_corner, dims)
Volume = np.log10(np.array(grid['density']).astype('float32').flat[:])
Volume -= Volume.min()
Volume /= Volume.max()
Volume *= 65536.0

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

ctf = "../data/colormaps/IceFire.cmap"

# opacity transfer function

otf = "../data/colormaps/Grayscale.orig.omap"

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
imagename = "PythonVolImageyt"

gvt.addFilm(filmname, wsize[0], wsize[1], imagename)

# renderer bits ...

rendername = "ytVolRender"

# these are the integer values of the two "types" they are
# enums in the C++. Hardwire here for domain schedul and ospray adapter
schedtype = 1
adaptertype = 5

gvt.addRenderer(rendername, adaptertype, schedtype, camname, filmname, True)

# render it

gvt.render(rendername)

# save the image

gvt.writeimage(rendername, imagename)
