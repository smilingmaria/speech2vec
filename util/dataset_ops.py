from glob import glob
import os
import pdb

import h5py
import numpy as np
from tqdm import tqdm

DATA_ROOT = './data/'
DEV_CLEAN = 'dev-clean/'
TEST_CLEAN = 'test-clean/'
FBANK = 'fbank/'
FBANK_DELTA = 'fbank_delta/'
DATA_NAME = 'data.h5'

CSV_EXT = ".csv"

def load_dataset(dataset, fbank_type, dir_root = './data/'):
    # dataset: train | dev | test
    if dataset == "dev":
        h5path = dir_root + DEV_CLEAN + fbank_type + DATA_NAME
    elif dataset == "test":
        h5path = dir_root + TEST_CLEAN + fbank_type + DATA_NAME
    else:
        raise Exception("No such dataset: {}".format(dataset))
    f = h5py.File(h5path,'r')
    return f['fbank'][:]

def create_dataset(path, fbank_type, ext = CSV_EXT):
    """
        Stores as array of objects( different size arrays )
    """
    print "Creating dataset at {0}...".format(path+fbank_type)
    print "Loading samples..." 
    fbank = []
    num_of_samples = len(glob(path+fbank_type+"*.csv"))
    for i in tqdm(range(1,num_of_samples+1,1)):
        filename = path + fbank_type + str(i) + ext

        arr = np.loadtxt(filename,delimiter=',',dtype='float32')
        fbank.append(arr)
    
    # Get Maxmimum Timestep
    print "Getting maximum timestep..."
    max_timestep = 0
    for i in tqdm(range(len(fbank))):
        if fbank[i].shape[0] > max_timestep:
            max_timestep = fbank[i].shape[0]

    # Pad Zeros & Stack
    print "Padding zeros"
    for i in tqdm(range(len(fbank))):
        ts, _ = fbank[i].shape
        fbank[i] = np.pad(fbank[i],((0,max_timestep-ts),(0,0)),'constant',constant_values=0.) 

    print "Stacking and rolling axis..."
    fbank = np.dstack(fbank)
    fbank = np.rollaxis(fbank,2,0)
    
    # Stack and save
    h5name = 'data.h5'
    print "Saving to {0}...".format(path+fbank_type+h5name)
    f = h5py.File(path+fbank_type+h5name,'w')
    f.create_dataset('fbank',data=fbank)
    f.close()
    print 'Done'
    print

# For matlab conversion
def save_to_csv(fbank,dir_name):
    assert dir_name.endswith("/")
    # Fbank is a numpy array  
    try: 
        os.makedirs(dir_name)
    except OSError:
        if not os.path.isdir(dir_name):
            raise

    for idx,arr in tqdm(enumerate(fbank)):
        fname = dir_name + str(idx) + ".csv" 
        mask = np.all(np.isnan(arr) | np.equal(arr, 0), axis=1)
        arr = arr[~mask]
        np.savetxt(fname,arr,delimiter=",") 

if __name__ == "__main__":
    fbank = load_dataset("dev","fbank/")
    
    save_to_csv(fbank,"tmp_csv/")



    



