from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext

import glob
import sys
import os
import numpy

compile_args = ['-std=c++11','-DGVT_BUILD_VOLUME']

try:
    embree_inc = os.sep.join([os.environ["embree_DIR"], "include"])
    embree_dir = os.environ["embree_DIR"]
    if os.path.exists(os.sep.join([embree_dir, 'lib'])):
        embree_lib = os.sep.join([embree_dir, 'lib'])
    elif os.path.exists(os.sep.join([embree_dir, 'lib64'])):
        embree_lib = os.sep.join([embree_dir, 'lib64'])
    else:
        raise RuntimeError("Cannot identify embree lib or lib64 dir, please "
                           "set the embree_DIR environment variable to the "
                           "location of the root of the embree installation.")
except KeyError:
    print("Please set the environment variable embree_DIR")
    sys.exit(1)


try:
    os.environ["gvt_DIR"]
    gvt_dir = os.environ["gvt_DIR"]
    gvt_inc = gvt_dir + "/include"
    gvt_api_inc = gvt_inc + "/gravit"
    gvt_lib = os.environ["gvt_DIR"] + "/lib"
except KeyError:
    print("Please set the environment variable gvt_DIR")
    sys.exit(1)

try:
    #os.environ["GregSpray_LIB_DIR"]
    os.environ["Galaxy_LIB_DIR"]
except KeyError:
    print("Please set the environment variable Galaxy_LIB_DIR")
    sys.exit(1)

try:
    os.environ["IceT_LIB_DIR"]
except KeyError:
    print("Please set the environment variable IceT_LIB_DIR")
    sys.exit(1)

try:
    mpi_dir = os.environ["MPI_DIR"]
    mpi_inc = os.sep.join([os.environ["MPI_DIR"], "include"])
    if os.path.exists(os.sep.join([mpi_dir, 'lib'])):
        mpi_lib = os.sep.join([mpi_dir, 'lib'])
    elif os.path.exists(os.sep.join([mpi_dir, 'lib64'])):
        mpi_lib = os.sep.join([mpi_dir, 'lib64'])
    else:
        raise RuntimeError("Cannot identify MPI lib or lib64 dir, please "
                           "set the MPI_DIR environment variable to the "
                           "location of the root of the MPI installation.")
    if glob.glob(os.sep.join([mpi_lib, 'libmpicxx*'])):
        mpi_cxx_lib = "mpicxx"
    else:
        mpi_cxx_lib = "mpi_cxx"
except KeyError:
    print("Please set the environment variable MPI_DIR")
    sys.exit(1)

mpi_mac = ""
if sys.platform == 'darwin':
    compile_args.append('-mmacosx-version-min=10.11')
    compile_args.append('-stdlib=libc++')
    mpi_mac = "-mt"


class build_ext_compiler_check(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == 'intel':
            for ext in self.extensions:
                ext.libraries.extend(["irc", "imf"])
        build_ext.build_extensions(self)


extensions = [
    Extension("gvt", ["src/gvt/gvt.pyx"],
              include_dirs=[
                 embree_inc,
                 gvt_inc,
                 gvt_api_inc,
                 mpi_inc,
                 numpy.get_include()],
              libraries=[
                 "gvtRender", "gvtCore",
                 "IceTGL", "IceTMPI", "IceTCore",
                 "embree3",
                 "mpi",
                 mpi_cxx_lib,
                 "ospray",
                 "gxy_data",
                 "gxy_framework",
                 "gxy_renderer",
                 "gxy_sampler",
              ],
              library_dirs=[
                 embree_lib,
                 mpi_lib,
                 gvt_lib,
                 os.environ["IceT_LIB_DIR"],
                 os.environ["ospray_LIB_DIR"],
                 os.environ["Galaxy_LIB_DIR"],
              ],
              language="c++",
              extra_compile_args=compile_args,
              extra_link_args=compile_args)
]

setup(
   name="gvt",
   cmdclass={"build_ext": build_ext_compiler_check},
   version="1.0.0",
   ext_modules=extensions
)
