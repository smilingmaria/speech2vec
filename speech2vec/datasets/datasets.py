from collections import namedtuple
import csv
import json
import os
import random

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

        self.setup()

        self.transform()

    def setup(self):
        self.X = self.fbank_delta
        self.y = to_categorical(self.labels)

    def transform(self):
        # Scale by mean and variance

        self._mean = self.X.mean(axis=0)
        self._std  = self.X.std(axis=0)

        self.normX = ( self.X - self._mean ) / self._std

    def inverse_transform(self, normed_arr):
        assert normed_arr.shape[1:] == self.X.shape[1:] 
        inv_arr = normed_arr * self._std + self._mean
        return inv_arr
         
    @property
    def shape(self):
        return self.X.shape

    def next_batch(self, norm = False, batch_size = 32, shuffle = True):
        if norm:
            X = self.normX
        else:
            X = self.X

        toadd = ( batch_size - X.shape[0] % batch_size ) % batch_size

        X = np.vstack([ X, X[:toadd] ] )

        assert X.shape[0] % batch_size == 0
        
        if shuffle:
            random.shuffle(X)
        
        for idx in range(0, X.shape[0], batch_size):
            x = X[idx:idx+batch_size]
            
            yield x

#####################
#  Datasets to call #
#####################

def dsp_hw2():
    h5_path = '../raw_data/dsp_hw2/dsp_hw2.h5' 
    return Dataset(h5_path)
