#
# cosmology_plus.py
#
# read and render the first two levels of the cosmology plus
# enzo dataset. There are actually 5 levels of refinement
# but the first two levels contain 37 grids. 
#
# This script is rough. It is only intended to test the python
# wrappers for amr grids. 
# to run this from inside the interpreter do
# exec(open('cosmology_plus.py').read())
#
# Import the required libs.
#
import gvt
import h5py
import os
from mpi4py import MPI
import numpy as np
#from vtk import vtkStructuredPointsReader, vtkStructuredPoints
#
# initialize GraviT
#
gvt.gvtInit()
#
# MPI business
#
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.size
#
print(" numprocs " + str(numprocs) + " rank " + str(rank))
#
#
# where are the data
#
data_dir = os.path.join(os.environ['WORK'],"Projects/GraviT/data/enzo_cosmology_plus")
gravit_dir = os.path.join(os.environ['WORK'],"Projects/GraviT")
# going to want to run this from the data directory
# so all the relative links work
imagedir = os.getcwd()
os.chdir(data_dir)
# input files 
volumefile = os.path.join(data_dir,"DD0046/DD0046.hierarchy.hdf5")
#ctffile = os.path.join(gravit_dir,"data/colormaps/Grayscale.orig.cmap")
#otffile = os.path.join(gravit_dir,"data/colormaps/Grayscale.orig.omap")
#ctffile = os.path.join(gravit_dir,"data/colormaps/CoolWarm.cmap")
otffile = os.path.join(gravit_dir,"data/colormaps/blue2cyan.omap")
#otffile = os.path.join(gravit_dir,"data/colormaps/ramp.omap")
#ctffile = os.path.join(gravit_dir,"data/colormaps/blue2cyan.cmap")
#ctffile = os.path.join(gravit_dir,"data/colormaps/IceFire.cmap")
#ctffile = os.path.join(gravit_dir,"data/colormaps/Jet.cmap")
ctffile = os.path.join(gravit_dir,"data/colormaps/coldhot.cmap")
#ctffile = os.path.join(gravit_dir,"data/colormaps/orange-5.cmap")
#otffile = os.path.join(gravit_dir,"data/colormaps/orange-5.omap")
#ctffile = os.path.join(gravit_dir,"data/colormaps/Balls.cmap")
#otffile = os.path.join(gravit_dir,"data/colormaps/Balls.omap")
#
root=h5py.File(volumefile)
# the number of domains is the number of grids in level 0
level0 = root['Level0']
numberofdomains = level0.attrs["NumberOfGrids"]
level0grids = list(level0.keys())
low_scalar = np.finfo('float32').max
high_scalar = np.finfo('float32').min
samplingrate = 1.0
k = 0
#for domain in range(1):
for domain in range(numberofdomains):
    level = 0 
    if(domain%numprocs == rank): # read the domain (grid)
        nodename = "enzo_cosmology_plus_domain_" + repr(domain)
#        print(" creating node " + nodename)
        gvt.createVolume(nodename,True)
        gridname = level0grids[domain]
        grid = level0[gridname]
        griddata = grid.get('GridData')
        density = griddata['Density']
        with density.astype('float32'):
            scalars = density[()]
        scalardims = np.array(scalars.shape,dtype=np.int32)
        low_scalar= min(low_scalar,scalars.min())
        high_scalar= max(high_scalar,scalars.max())
        #dimensions = grid['GridDimension'].value
        startindex = grid['GridStartIndex'][()]
        endindex = grid['GridEndIndex'][()]
        dimensions = (endindex - startindex)+1 
        #dimensions = scalardims
        origin = grid['GridGlobalPosition'][()]
        left = grid['GridLeftEdge'][()]
        right = grid['GridRightEdge'][()]
        spacing = (right - left)/(dimensions)
        right = left + spacing*(dimensions)
        bounds = np.array([left[0],right[0],left[1],right[1],left[2],right[2]])
        # stuff the level grid full
#        print("\tdims "+repr(dimensions[:]))
#        print("\tsdims "+repr(scalardims[:]))
#        print("\tleft " + repr(left[:]))
#        print("\tspacing " + repr(spacing))
#        print("\tsampling " + repr(samplingrate))
#        print("\tbounds " + repr(bounds))
        #fltptr = scalars.flatten()
        fltptr = np.ravel(scalars,order='C')
#        print("\tfloats " + repr(fltptr[0]) + " " +repr(fltptr[1] ))
#        print("level " + repr(level) + " gridname " + gridname +" nodename "+ nodename)
        gvt.addVolumeSamples(nodename,fltptr.astype(np.float32),dimensions.astype(np.int32),left.astype(np.float32),spacing.astype(np.float32),samplingrate,bounds.astype(np.float64))
        # grab the subgrids or daughters of this grid
        daughtergrids = grid['DaughterGrids']
        dglist = list(daughtergrids.keys())
        numsubs = len(dglist)
        #for l in range(0):
        for dgname in daughtergrids.keys():
            #dgname = dglist[l]
            level = 1
            k = k + 1
            grid = daughtergrids[dgname]
            griddata = grid.get('GridData')
            density = griddata['Density']
            with density.astype('float32'):
                scalars = density[()]
            scalardims = np.array(scalars.shape,dtype=np.int32) -1
            low_scalar= min(low_scalar,scalars.min())
            high_scalar= max(high_scalar,scalars.max())
            #dimensions = grid['GridDimension'].value
            startindex = grid['GridStartIndex'][()]
            endindex = grid['GridEndIndex'][()]
            dimensions = endindex - startindex 
            origin = grid['GridGlobalPosition'][()]
            left = grid['GridLeftEdge'][()]
            right = grid['GridRightEdge'][()]
            bounds = np.array([left[0],right[0],left[1],right[1],left[2],right[2]])
            spacing = (right - left)/(endindex-startindex +1)
#            print("\t"+dgname)
#            print("\t\tdims "+repr(dimensions[:]))
#            print("\t\tleft " + repr(left[:]))
#            print("\t\tright " + repr(right[:]))
#            print("\t\tspacing " + repr(spacing))
#            print("\tbounds " + repr(bounds))
            fltptr = scalars.flatten()
#            print("level "+repr(level)+" gridname "+dgname+" nodename "+nodename)
            gvt.addAmrSubgrid(nodename,k,level,fltptr.astype(np.float32),dimensions.astype(np.int32),left.astype(np.float32),spacing.astype(np.float32))
        print(" add transfer functions " + nodename)
        print(" ctffile : " + ctffile)
        print(" otffile : " + otffile)
        low_scalar = 0.10
        high_scalar = 42.0
        print(" scalar range : " + repr(low_scalar) + " " + repr(high_scalar))
        gvt.addVolumeTransferFunctions(nodename,ctffile,otffile,low_scalar,high_scalar)
        # add an instance for this level 0 grid
        mf = np.identity(4,dtype=np.float32).flatten()
        myinstance = "inst" + repr(domain)
        gvt.addInstance(myinstance,nodename,mf)
# and now camera etc.
#
eyept = np.array([2.0,2.0,2.0],dtype=np.float32)
focus = np.array([0.4,0.6,0.5],dtype=np.float32)
fov = 10.0*np.pi/180.0
upVector = np.array([0.0,1.0,0.0],dtype=np.float32)
rayMaxDepth = 1
raysamples = 1
jitterWindowSize = 0.5
camname = "conecam"
gvt.addCamera(camname,eyept,focus,upVector,fov,rayMaxDepth,raysamples,jitterWindowSize)
#film
wsize = np.array([640,640],dtype=np.int32)
filmname = "conefilm"
imagename = "EnzoImage"
gvt.addFilm(filmname,wsize[0],wsize[1],imagename)
#render
rendername = "EnzoVolRenderer"
schedtype = 1
adaptertype = 6
gvt.addRenderer(rendername,adaptertype,schedtype,camname,filmname,True)
gvt.render(rendername)
os.chdir(imagedir)
gvt.writeimage(rendername,imagename)
