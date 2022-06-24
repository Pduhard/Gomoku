#!/bin/sh

# Dependencies
pip3 uninstall numpy
pip3 install pygame matplotlib cffi numba scipy

# Compile the Library
make -C GomokuLib
