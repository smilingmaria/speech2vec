import csv
from glob import glob
import os
import sys

import h5py
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

# Will need to find an alternative!

def create_data_h5(dataset_root):
    """
        Reads all csv in direcotry and in order
        Stores as array of objects( different size arrays )
    """
    print "Creating dataset at {0}...".format(dataset_root)

    # Read fbank
    print "Reading fbank..."
    fbank = read_csv_to_arr( dataset_root + 'fbank/' )

    # Read fbank_delta
    print "Reading fbank_delta..."
    fbank_delta = read_csv_to_arr( dataset_root + 'fbank_delta/' )

    # read yphase
    print "Reading yphase"
    yphase = read_csv_to_arr( dataset_root + 'yphase/' )

    # read wav
    print "Reading wav..."
    wav = read_wav_to_arr( dataset_root + 'wavs/')

    # read labels
    print "Reading labels..."
    labels, genders = read_labels_to_arr( dataset_root + 'feature.map' )

    # save to data_h5
    save_to_data_h5( dataset_root + 'data.h5', fbank, fbank_delta, yphase, wav, labels, genders )

def read_csv_to_arr(path):
    fbank = []
    num_of_samples = len(glob(path+"*.csv"))
    for i in range(1,num_of_samples+1,1):
        filename = path + str(i) + '.csv'
        arr = np.loadtxt(filename,delimiter=',',dtype='float32')
        fbank.append(arr)

    # Get Maxmimum Timestep
    max_timestep = 0
    for i in range(len(fbank)):
        if fbank[i].shape[0] > max_timestep:
            max_timestep = fbank[i].shape[0]

    # Pad Zeros & Stack
    for i in range(len(fbank)):
        ts, _ = fbank[i].shape
        pad_front = (max_timestep - ts) / 2
        pad_back  = (max_timestep - ts+1) / 2
        fbank[i] = np.pad(fbank[i],((pad_front,pad_back),(0,0)),'constant',constant_values=0.)

        assert fbank[i].shape[0] == max_timestep, "Padding failed"

    fbank = np.dstack(fbank)
    fbank = np.rollaxis(fbank,2,0)

    return fbank

def read_wav_to_arr(path):
    wav = []
    for wavname in sorted( glob(path + '*.wav') ):
        w = wavfile.read(wavname)
        sr, arr = w
        arr = np.array(arr,dtype='float32')
        wav.append(arr)

    from keras.preprocessing import sequence
    wav = sequence.pad_sequences( wav, padding='post', dtype='float32' )
    # Add one more dimension since wav is original 2D
    wav = wav[...,None]
    return wav

def read_labels_to_arr(path_to_map):
    labels = []
    genders = []
    with open( path_to_map,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(int(row[2]))
            genders.append(int(row[3]))

    labels = np.array(labels)
    genders = np.array(genders)

    return labels, genders


def save_to_data_h5(path, fbank, fbank_delta, yphase, wav, labels, genders):
    print "Saving to {0}...".format(path)
    f = h5py.File(path,'w')
    f.create_dataset('fbank',data=fbank)
    f.create_dataset('fbank_delta',data=fbank_delta)
    f.create_dataset('yphase',data=yphase)
    f.create_dataset('wav',data=wav)
    f.create_dataset('labels',data=labels)
    f.create_dataset('genders',data=labels)
    f.close()

if __name__ == "__main__":
    project_root = os.path.abspath('..')
    if len(sys.argv) != 2:
        raise ValueError("Specify data set path!")
    dataset_dir = project_root + '/' + sys.argv[1]
    dataset_dir = sys.argv[1]
    if not os.path.exists( dataset_dir + 'data.h5'):
        create_data_h5(dataset_dir)
    else:
        print("Data set already exists for {}".format(dataset_dir))


