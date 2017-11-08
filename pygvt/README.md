# README #

pyGVT - Python wrapper for GraviT

### Dependencies

* Embree 2.15+
* IceT
* GraviT
* MPI

Add all lib Dependencies paths to LD_LIBRARY_PATH

### Build package

Edit setenv.sh and change the paths according to your Dependencies install

#### [Optional] Create a virtual env to tests
```
pip install virtualenv
virtualenv ~/.pve/pyGVT
source ~/.pve/pyGVT/bin/active
```

#### Install
```
source setenv.sh
python setup.py install
```

### Test

```
python test.py
display simple.ppm
```
```
source test_example_reader.sh
display bunny.ppm
display wavelet.ppm
display block0.ppm
```
