import os

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import h5py
import numpy as np
from tqdm import tqdm

from utils import makedir

# Function that includes all other functions for saving
def save_reconstruction( sess, model, minloss_modelname, save_dir, dataset ):
    makedir(save_dir)
    batch_size = model.batch_input_shape[0]

    X_rec = model.reconstruct(sess,dataset.next_batch(batch_size=batch_size, shuffle = False))
    code  = model.encode(sess,dataset.next_batch(batch_size=batch_size, shuffle = False))

    X_rec = dataset.fit_X_shape(X_rec)
    code  = dataset.fit_X_shape(code)

    h5_path = save_dir + minloss_modelname + '.h5'

    print "Saving feature and yphase to %s" % h5_path
    save_h5( h5_path, X_rec, code )
    """
    feature_path = save_dir + dataset.data_type + '/'
    print "Saving feature to %s" % feature_path
    save_to_csv( feat, feature_path)

    yphase_path = save_dir + 'yphase/'
    print "Saving yphase to %s" % yphase_path
    save_to_csv( phase, yphase_path )
    """

def save_h5(h5_path, recX, code):
    with h5py.File(h5_path, 'w') as h5f:
        h5f.create_dataset("recX",data=recX)
        #h5f.create_dataset("yphase",data=yphase)
        h5f.create_dataset("code",data=code)

def save_to_csv( arr,dir_name):
    assert dir_name.endswith("/")
    makedir(dir_name)

    for idx,arr in tqdm(enumerate(arr)):
        fname = dir_name + str(idx+1) + ".csv"
        # Convert Nan to zeros
        arr = np.nan_to_num(arr)
        # Remove rows that are all nans or all zeros
        #mask = np.all(np.isnan(arr) | np.equal(arr, 0), axis=1)
        #arr = arr[~mask]
        np.savetxt(fname,arr,delimiter=",")
