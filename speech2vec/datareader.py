import logging
import os
import random

import h5py
import numpy as np

# Took from keras
def to_categorial(y, nb_classes=None):
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

class DataReader(object):
    def __init__(self, h5_path, data_type='fbank_delta', with_yphase = False, batch_size = 32):
        # H5 File handler
        self.h5_handle = h5py.File(h5_path,'r')

        # Data type
        self._data_type = data_type

        # With or without yphase
        self.with_yphase = with_yphase

        # batch size for generator
        self.batch_size = batch_size


    def __del__(self):
        self.h5_handle.close()

    @property
    def name(self):
        namestring = self._data_type
        if self.with_yphase:
            namestring += '_yphase'
        return namestring

    @property
    def data_type(self):
        return self._data_type

    @property
    def shape(self):
        if self.with_yphase:
            nb_samples, timestep, feature_size = self.h5_handle[ self.data_type ].shape
            _, _, yphase_size = self.h5_handle['yphase'].shape
            return (nb_samples, timestep, feature_size + yphase_size)
        return self.h5_handle[ self.data_type ].shape

    @property
    def nb_samples(self):
        return self.shape[0]

    def next_batch_generator(self, shuffle = True):
        sample_idx_list = range(self.nb_samples)

        while True:
            if shuffle:
                random.shuffle(sample_idx_list)

            for idx in range(0, self.nb_samples,self.batch_size):
                batch_idx_list = sample_idx_list[idx:idx+self.batch_size]
                batch_idx_list.sort() # Increasing order for h5py dataset

                if self.with_yphase:
                    batch_feature = self.h5_handle[ self.data_type ][ batch_idx_list ]
                    batch_yphase = self.h5_handle['yphase'][ batch_idx_list ]

                    batch_X = np.dstack([ batch_feature, batch_yphase ])
                else:
                    batch_X = self.h5_handle[ self.data_type ][ batch_idx_list ]

                yield batch_X, batch_X
