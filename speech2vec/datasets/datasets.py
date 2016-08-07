from collections import namedtuple
import csv
import json
import os

import h5py
import numpy as np

from np_utils import to_categorical

class Dataset(object):
    def __init__(self, h5_path, fbank_type = 'fbank_delta'):
        """
            Fbank delta only at the moment
        """
        h5f = h5py.File(h5_path,'r')
        self.fbank_delta = h5f[ fbank_type ][:]
        self.labels = h5f['labels'][:]

        self.normalize()

    def normalize(self):
        """
            Provides normalization, and number to categorial labels
        """
        self.X = self.fbank_delta
        self.y = to_categorical(self.labels)

    @property
    def shape(self):
        return self.fbank_delta.shape

    def next_batch(self, batch_size = 32, reverse = False):
        X = self.fbank_delta

        toadd = ( batch_size - X.shape[0] % batch_size ) % batch_size

        X = np.vstack([ X, X[:toadd] ] )

        assert X.shape[0] % batch_size == 0

        for idx in range(0, X.shape[0], batch_size):
            x = X[idx:idx+batch_size]

            if reverse:
                yield x[...,::-1], x
            else:
                yield x, x

#####################
#  Datasets to call #
#####################

def dsp_hw2():
    h5_path = '../raw_data/dsp_hw2/dsp_hw2.h5' 
    return Dataset(h5_path)
