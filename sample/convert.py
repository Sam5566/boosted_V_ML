from __future__ import absolute_import

import tensorflow as tf
from writeTFR import *
import sys
import os

def main():
    filelist = sys.argv[1:]
    create_TFRecord(filelist)

    #for fname in filelist:
    #    os.remove(fname)

if __name__ == '__main__':
    main()
