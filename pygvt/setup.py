from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import sys
import os
import numpy

compile_args = ['-std=c++11']

try:
   embree_inc = os.environ["embree_DIR"] + "/include"
   embree_lib = os.environ["embree_DIR"] + ("/lib" if os.path.exists(os.environ["embree_DIR"]+"/lib") else "/lib64")

except KeyError:
   print("Please set the environment variable embree_DIR")
   sys.exit(1)


try:
   os.environ["Boost_DIR"]
   boost_inc = os.environ["Boost_DIR"] + "/include"
   boost_lib = os.environ["Boost_DIR"] + ("/lib" if os.path.exists(os.environ["Boost_DIR"]+"/lib") else "/lib64")

except KeyError:
   print("Please set the environment variable Boost_DIR")
   sys.exit(1)


try:
    os.environ["gvt_DIR"]
    gvt_inc = os.environ["gvt_DIR"] + "/include"
    gvt_lib = os.environ["gvt_DIR"] + "/lib"
except KeyError:
    print("Please set the environment variable gvt_DIR")
    sys.exit(1)

try:
    os.environ["IceT_LIB_DIR"]
except KeyError:
    print("Please set the environment variable IceT_LIB_DIR")
    sys.exit(1)

try:
   os.environ["MPI_DIR"]
   mpi_inc = os.environ["MPI_DIR"] + "/include"
   mpi_lib = os.environ["MPI_DIR"] + ("/lib" if os.path.exists(os.environ["MPI_DIR"]+"/lib") else "/lib64")
except KeyError:
   print("Please set the environment variable MPI_DIR")
   sys.exit(1)

print(compile_args)

mpi_mac=""
if sys.platform == 'darwin':
    compile_args.append('-mmacosx-version-min=10.11')
    compile_args.append('-stdlib=libc++')
    mpi_mac="-mt"

extensions = [
    Extension("gvt",["src/gvt/gvt.pyx"],
        include_dirs = [
            embree_inc,
            boost_inc,
            gvt_inc,
            mpi_inc,
            # os.environ["embree_DIR"] + "/include",
            # os.environ["MPI_DIR"] + "/include",
            # os.environ["Boost_DIR"] + "/include",
            # os.environ["gvt_DIR"]+"/include",
            numpy.get_include()],
        libraries = [
        "gvtRender","gvtCore",
        "plyloader",
        "IceTGL","IceTMPI","IceTCore",
        "embree",
        "boost_system"+mpi_mac,
        "mpi",
        "mpicxx",
        "irc",
        "imf"
        ],
        library_dirs = [
            embree_lib,
            mpi_lib,
            boost_lib,
            gvt_lib,
            os.environ["IceT_LIB_DIR"],
            ],
        language="c++",
        extra_compile_args=compile_args, # + mpi_compile_args,
        extra_link_args=compile_args # + mpi_link_args
        )
]
setup(
    name = "gvt",
    cmdclass = {"build_ext": build_ext},
    version="1.0.0",
    ext_modules = extensions
    # ext_modules = cythonize(extensions)
)
