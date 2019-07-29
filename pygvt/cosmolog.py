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
def volinput(level0,domain,com,rank):
    k = 0
    print(f'rank {rank} reading domain {domain}')
    nodename = "enzo_cosmology_plus_domain_" + repr(domain)
    gvt.createVolume(nodename,True)
    #gridname = level0grids[domain]
    #grid = level0[gridname]
    #level0grids = list(level0.keys())
    #grid = level0[list(level0grids[domain]]
    grid = level0[list(level0.keys())[domain]]
    griddata = grid.get('GridData')
    density = griddata['Density']
    with density.astype('float32'):
        scalars = np.log10(density[()])
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
    #fltptr = scalars.flatten()
    fltptr = np.ravel(scalars,order='C')
    gvt.addVolumeSamples(nodename,fltptr.astype(np.float32), \
            dimensions.astype(np.int32),left.astype(np.float32),\
            spacing.astype(np.float32),samplingrate,bounds.astype(np.float64))
    # grab the subgrids or daughters of this grid
    daughtergrids = grid['DaughterGrids']
    dglist = list(daughtergrids.keys())
    numsubs = len(dglist)
    for dgname in daughtergrids.keys():
        #dgname = dglist[l]
        level = 1
        k = k + 1
        grid = daughtergrids[dgname]
        griddata = grid.get('GridData')
        density = griddata['Density']
        with density.astype('float32'):
            scalars = np.log10(density[()])
        scalardims = np.array(scalars.shape,dtype=np.int32) -1
        low_scalar= min(low_scalar,scalars.min())
        high_scalar= max(high_scalar,scalars.max())
        startindex = grid['GridStartIndex'][()]
        endindex = grid['GridEndIndex'][()]
        dimensions = endindex - startindex 
        origin = grid['GridGlobalPosition'][()]
        left = grid['GridLeftEdge'][()]
        right = grid['GridRightEdge'][()]
        bounds = np.array([left[0],right[0],left[1],right[1],left[2],right[2]])
        spacing = (right - left)/(endindex-startindex +1)
        fltptr = scalars.flatten()
        gvt.addAmrSubgrid(nodename,k,level,fltptr.astype(np.float32),\
                dimensions.astype(np.int32),left.astype(np.float32),\
                spacing.astype(np.float32))
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
#print(" numprocs " + str(numprocs) + " rank " + str(rank))
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
#otffile = os.path.join(gravit_dir,"data/colormaps/blue2cyan.omap")
otffile = os.path.join(gravit_dir,"data/colormaps/tenspikes.omap")
#otffile = os.path.join(gravit_dir,"data/colormaps/ramp.omap")
#ctffile = os.path.join(gravit_dir,"data/colormaps/blue2cyan.cmap")
#ctffile = os.path.join(gravit_dir,"data/colormaps/blueredyellowwhite.cmap")
#ctffile = os.path.join(gravit_dir,"data/colormaps/gist_stern.cmap")
#ctffile = os.path.join(gravit_dir,"data/colormaps/Jet.cmap")
#ctffile = os.path.join(gravit_dir,"data/colormaps/coldhot.cmap")
#ctffile = os.path.join(gravit_dir,"data/colormaps/orange-5.cmap")
ctffile = os.path.join(gravit_dir,"data/colormaps/RdBu_r.cmap")
#otffile = os.path.join(gravit_dir,"data/colormaps/orange-5.omap")
#ctffile = os.path.join(gravit_dir,"data/colormaps/Balls.cmap")
#otffile = os.path.join(gravit_dir,"data/colormaps/Balls.omap")
#
# the number of domains is the number of grids in level 0
root=h5py.File(volumefile,'r')
level0 = root['Level0']
numberofdomains = level0.attrs["NumberOfGrids"]
level0grids = list(level0.keys())
low_scalar = np.finfo('float32').max
high_scalar = np.finfo('float32').min
samplingrate = 1.0
#for domain in range(1):
for domain in range(numberofdomains):
    level = 0 
    if(domain%numprocs == rank): # read the domain (grid)
        k = 0
        #volinput(level0grids, domain,comm,rank)
        nodename = "enzo_cosmology_plus_domain_" + repr(domain)
        gvt.createVolume(nodename,True)
        gridname = level0grids[domain]
        grid = level0[gridname]
        griddata = grid.get('GridData')
        density = griddata['Density']
        with density.astype('float32'):
            scalars = np.log10(density[()])
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
        #fltptr = scalars.flatten()
        fltptr = np.ravel(scalars,order='C')
        gvt.addVolumeSamples(nodename,fltptr.astype(np.float32),\
                dimensions.astype(np.int32),left.astype(np.float32),\
                spacing.astype(np.float32),samplingrate,bounds.astype(np.float64))
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
                scalars = np.log10(density[()])
            scalardims = np.array(scalars.shape,dtype=np.int32) -1
            low_scalar= min(low_scalar,scalars.min())
            high_scalar= max(high_scalar,scalars.max())
            startindex = grid['GridStartIndex'][()]
            endindex = grid['GridEndIndex'][()]
            dimensions = endindex - startindex 
            origin = grid['GridGlobalPosition'][()]
            left = grid['GridLeftEdge'][()]
            right = grid['GridRightEdge'][()]
            bounds = np.array([left[0],right[0],left[1],right[1],left[2],right[2]])
            spacing = (right - left)/(endindex-startindex +1)
            fltptr = scalars.flatten()
            gvt.addAmrSubgrid(nodename,k,level,fltptr.astype(np.float32),\
                    dimensions.astype(np.int32),left.astype(np.float32),\
                    spacing.astype(np.float32))
        #print(" add transfer functions " + nodename)
        #print(" ctffile : " + ctffile)
        #print(" otffile : " + otffile)
        #low_scalar = 0.10
        #high_scalar = 42.0
        high_scalar = np.log10(100.0)
        #print(" scalar range : " + repr(low_scalar) + " " + repr(high_scalar))
        gvt.addVolumeTransferFunctions(nodename,ctffile,otffile,low_scalar,high_scalar)
gvt.gvtsync()
mf = np.identity(4,dtype=np.float32).flatten()
for domain in range(numberofdomains):
    nodename = "enzo_cosmology_plus_domain_" + repr(domain)
    # add an instance for this level 0 grid
    myinstance = "inst" + repr(domain)
    gvt.addInstance(myinstance,nodename,mf)
gvt.gvtsync()
# and now camera etc.
#
eyept = np.array([2.5,2.5,2.5],dtype=np.float32)
#focus = np.array([0.65,0.7,0.6],dtype=np.float32)
focus = np.array([0.5,0.5,0.5],dtype=np.float32)
#focus = np.array([0.461,0.211,0.570],dtype=np.float32)
fov = 30.0*np.pi/180.0
upVector = np.array([0.0,1.0,0.0],dtype=np.float32)
rayMaxDepth = 1
raysamples = 1
jitterWindowSize = 0.5
camname = "conecam"
gvt.addCamera(camname,eyept,focus,upVector,fov,rayMaxDepth,raysamples,jitterWindowSize)
#film
wsize = np.array([512,512],dtype=np.int32)
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
#print(f'move the camera and render again {eyept}')
#eyept[2] = 1.5
#gvt.modifyCamera(camname,eyept,focus,upVector,fov)
#gvt.render(rendername)
#imagename = imagename + str(0)
#print(imagename)
#gvt.writeimage(rendername,imagename)
