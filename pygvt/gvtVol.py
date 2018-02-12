#
# gvtVol.py 
#
# This application tests the python interface for the volume rendering parts of the
# GraviT API. It is intended to be run on exactly 2 mpi ranks and will halt if
# you try to run it on any other number of ranks.
# The code is hardwired to split the volume into a top and bottom part and to
# use the domain schedule and ospray adapter. Messing with those settings will
# have unpredictable results.
#
import gvt
from mpi4py import MPI
import numpy as np

# A python implementation of the GraviT volume renderer

gvt.gvtInit()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.size
if numprocs != 2: # abort if wrong number of procs
    print("Run gvtVol.py with exactly 2 ranks not " + str(numprocs))
    exit()
# Read some data

dims = (256,256,256)
halfdims = (256,256,128)
Volume = np.fromfile("../data/vol/1000.int256",dtype=np.int32).astype(np.float32)
npoints = Volume.size
topvol = Volume[int(npoints/2):npoints-1]
bottomvol = Volume[0:int(npoints/2-1)]

# Some volume metadata

# Deltas - cell spacing

deltas = np.array([1.0,1.0,1.0],dtype=np.float32)

# Counts - number of points in each coordinate direction

counts = np.array(halfdims,dtype=np.int32)

# Origin - origin of the volume in world space

origintop = np.array([0.0,0.0,127.0],dtype=np.float32)
originbottom = np.array([0.0,0.0,0.0],dtype=np.float32)

# samplingrate - rate of the sampling

samplingrate = 1.0 

# color transfer function

ctf = "../data/colormaps/Grayscale.orig.cmap"

# opacity transfer function

otf = "../data/colormaps/Grayscale.orig.cmap"

# name of the volume node on each rank

volnodename = "1000.int256" + str(rank)

# min and max values of the data.

low = 0.0
high = 65536.0

# make da volume

gvt.createVolume(volnodename)
gvt.addVolumeTransferFunctions(volnodename,ctf,otf,low,high)
if rank == 0:
    gvt.addVolumeSamples(volnodename,bottomvol,counts,originbottom,deltas,samplingrate)
else:
    gvt.addVolumeSamples(volnodename,topvol,counts,origintop,deltas,samplingrate)

# add an instance. 
# An instance needs a transformation matrix. All coords of the volume data
# are in world coords so we give the identity matrix.

mf = np.identity(4,dtype=np.float32).flatten()
myinstance = "inst" + str(rank)
gvt.addInstance(myinstance,volnodename,mf)

# Set up camera and film. 

eyept = np.array([512.0,512.0,512.0],dtype=np.float32)
focus = np.array([127.5,127.5,127.5],dtype=np.float32)
fov = 30.0*np.pi/180.0
upVector = np.array([0.0,0.0,1.0],dtype=np.float32)
rayMaxDepth = 1
raySamples = 1
jitterWindowSize = 0.5
camname = "volCam"

gvt.addCamera(camname,eyept,focus,upVector,fov,rayMaxDepth,raySamples,jitterWindowSize)

# set up film.

wsize = np.array([512,512],dtype=np.int32)
filmname = "volFilm"
imagename = "PythonVolImage"

gvt.addFilm(filmname,wsize[0],wsize[1],imagename)

# renderer bits ...

rendername = "PythonVolRenderer"

# these are the integer values of the two "types" they are
# enums in the C++. Hardwire here for domain schedul and ospray adapter
schedtype = 1
adaptertype = 5

gvt.addRenderer(rendername,adaptertype,schedtype,camname,filmname,True)

# render it

gvt.render(rendername)

# save the image

gvt.writeimage(rendername,imagename)
