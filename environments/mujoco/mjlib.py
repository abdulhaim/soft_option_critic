from ctypes import *
import os

osp = os.path

libfile = osp.join(osp.expanduser('~'), '.mujoco/mjpro131/bin/libmujoco131.so')

mjlib = cdll.LoadLibrary(libfile)

