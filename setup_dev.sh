#!/bin/sh

pip3 install numpy==1.18.0
pip3 install numba==0.55
pip3 install pygame==1.23.0
pip3 install matplotlib
pip3 install cffi

#pip3 install pygame matplotlib cffi numba

make -C GomokuLib
