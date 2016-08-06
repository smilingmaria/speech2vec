from collections import namedtuple
import csv
import json
import os

import h5py
import numpy as np

class Dataset(object):
    def __init__(self, h5_path):
        h5f = h5py.File(h5_path,'r')
        self.fbank = h5f['fbank'][:]
        self.fbank_delta = h5f['fbank_delta'][:]
        self.labels = h5f['labels'][:]


def dsp_hw2():
    h5_path = '../raw_data/dsp_hw2/dsp_hw2.h5' 
    return Dataset(h5_path)
