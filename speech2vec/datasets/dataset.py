import os

import h5py
import numpy as np

import sklearn.utils
from scaler import InstanceWise, FeatureWise

class Dataset(object):
    """
        Every dataset should have the following methods:
        - next_batch: iterates through next batch of self._X, self._y
    """
    def __init__(self, h5_path, data_type):
        with h5py.File(h5_path,'r') as h5f:
            X = h5f[ data_type ][:]
            labels = h5f['labels'][:]
       
        self._X = X
        self._y = to_categorical(labels)
       
        self._normX = self.fit_transform(self._X) 
    
    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def normX(self):
        return self._normX
    
    def next_batch(self, norm = False, batch_size = 32, shuffle = True):
        if norm:
            X = self.normX
        else:
            X = self.X

        Y = self.y

        toadd = ( batch_size - X.shape[0] % batch_size ) % batch_size

        X = np.vstack([ X, X[:toadd] ] )
        Y = np.vstack([ y, y[:toadd] ] ) 
        
        assert X.shape[0] % batch_size == 0
        
        if shuffle:
            X, Y = sklearn.utils.shuffle(X, Y, random_state=0)
        
        for idx in range(0, X.shape[0], batch_size):
            x = X[idx:idx+batch_size]
            y = Y[idx:idx+batch_size]
            yield x, y

class Acoustic(Dataset, InstanceWise):
    def __init__(self, h5_path, data_type):
        super(Acoustic, self).__init__(h5_path,data_type)

class Spectral(Dataset, InstanceWise):
    def __init__(self, h5_path, data_type):
        super(Spectral, self).__init__(h5_path,data_type)


###################
#  Dataset Utils  #
################### 

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

###########################
#                         #
#     Datasets to call    #
#                         #
###########################

project_root = os.path.abspath('../../')
raw_data_dir = project_root + '/raw_data/'

def dsp_hw2(data_type='fbank_delta'):
    
    h5_path = raw_data_dir + 'dsp_hw2/dsp_hw2.h5' 

    if data_type == 'fbank' or data_type == 'fbank_delta':
        dataset_class = Spectral
    elif data_type == 'wav':
        dataset_class = Acoustic
    else:
        raise ValueError("Unknown data type %s",data_type)

    return dataset_class( h5_path, data_type)
