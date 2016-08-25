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

        self._data_type = data_type
        with h5py.File(h5_path,'r') as h5f:
            feature = h5f[ data_type ][:]
            yphase = h5f['yphase'][:]
            labels = h5f['labels'][:]

        self._feature = feature
        # Has no use for wav
        self._yphase = yphase
        self._y = to_categorical(labels)

        # Normalization
        self._normfeature = self.fit_transform(self._feature)
        self._normyphase  = self.fit_transform(self._yphase)

    @property
    def data_type(self):
        return self._data_type

    @property
    def X(self):
        return np.dstack([ self._feature, self._yphase ])

    @property
    def normX(self):
        return np.dstack([ self._normfeature, self._normyphase ])

    @property
    def y(self):
        return self._y

    def next_batch(self, batch_size = 32, norm = False, shuffle = False):
        # Concatenate X and yphase for unified input
        if norm:
            X = self.normX
        else:
            X = self.X

        Y = self.y

        # Add samples according to batch_size
        toadd  = ( batch_size - X.shape[0] % batch_size ) % batch_size
        X      = np.vstack([ X, X[:toadd] ] )
        Y      = np.vstack([ Y, Y[:toadd] ] )

        assert X.shape[0] % batch_size == 0

        if shuffle:
            X, Y = sklearn.utils.shuffle(X, Y, random_state=0)

        for idx in range(0, X.shape[0], batch_size):
            x = X[idx:idx+batch_size]
            y = Y[idx:idx+batch_size]

            yield x, y

    def fit_X_shape(self, X_pred):
        sample = self.X.shape[0]
        return X_pred[:sample]

    def split_X(self, X_rec):
        X_rec = self.fit_X_shape(X_rec)

        feature_length = self._feature.shape[-1]
        yphase_length  = self._yphase.shape[-1]

        assert feature_length + yphase_length == X_rec.shape[-1]

        feature = X_rec[:,:,:feature_length]
        yphase  = X_rec[:,:,feature_length:]

        return feature, yphase

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

cur_dir = os.path.abspath(os.path.dirname(__file__))

project_root = cur_dir + '/../../'
raw_data_dir = project_root + 'raw_data/'

def dsp_hw2(data_type='fbank_delta'):

    h5_path = raw_data_dir + 'dsp_hw2/data.h5'

    if data_type == 'fbank' or data_type == 'fbank_delta':
        dataset_class = Spectral
    elif data_type == 'wav':
        dataset_class = Acoustic
    else:
        raise ValueError("Unknown data type %s",data_type)

    return dataset_class( h5_path, data_type)
