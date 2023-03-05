from __future__ import absolute_import

import tensorflow as tf
from writeTFR import *
import sys
import os

def main():
    filelist = sys.argv[1:]
    idlist = create_TFRecord(filelist)
    create_npy(filelist, set_idlist=idlist) # The inherit of idlist can make the same arrange of sample

    #for fname in filelist:
    #    os.remove(fname)

if __name__ == '__main__':
    main()
