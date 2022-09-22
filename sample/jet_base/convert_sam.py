from __future__ import absolute_import

import tensorflow as tf
from writeTFR import *
from shuffle import *
import sys
import os

def main():
    filelist = sys.argv[1:]
    Ndata, shuffled_dataName = npy_shuffle(filelist)
    create_TFRecord(filelist, Ndata, shuffled_dataName)

    #for fname in filelist:
    #    os.remove(fname)

if __name__ == '__main__':
    main()