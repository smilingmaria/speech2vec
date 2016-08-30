import logging
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath('..'))

from speech2vec.datareader import DataReader
from speech2vec.models import CNNAutoencoder

def run_training( model_args, training_args ):
    ####################
    #    Read Params   #
    ####################

    # Model Args
    hidden_dim        = model_args.get('hidden_dim',128)
    encode_dim        = model_args.get('encode_dim',2)
    nb_filters        = model_args.get('nb_filters',32)
    nb_conv           = model_args.get('nb_cols',3)
    depth             = model_args.get('depth',(1,1))
    dropout_keep_prob = model_args.get('dropout_keep_prob',0.8)

    # Training data args
    h5_path           = training_args.get('h5_path')
    data_type         = training_args.get('data_type','fbank_delta')
    with_yphase       = training_args.get('with_yphase', False)

    # Training args
    nb_epochs         = training_args.get('nb_epochs',10)
    batch_size        = training_args.get('batch_size',16)
    eval_epoch        = training_args.get('eval_epoch',100)
    result_root       = training_args.get('result_root','./result_root/')

    ####################
    #   Read & Train   #
    ####################

    # Reader
    reader = DataReader( h5_path     = h5_path,
                         data_type   = data_type,
                         with_yphase = with_yphase,
                         batch_size  = batch_size )

    # Build Model
    model = CNNAutoencoder( data_reader = reader,
                            hidden_dim = hidden_dim,
                            nb_filters = nb_filters,
                            nb_conv = nb_conv,
                            encode_dim = encode_dim,
                            depth = depth,
                            dropout_keep_prob = dropout_keep_prob )
    model.build_graph()

    # Start training
    model.train( nb_epochs=nb_epochs,
                 result_root=result_root,
                 eval_epoch=eval_epoch)

if __name__ == "__main__":

    training_args = {
            'h5_path': '/home/ubuntu/speech2vec/raw_data/dsp_hw2/data.h5',
            'data_type': 'fbank_delta',
            'with_yphase': True,
            'nb_epochs': 1000,
            'batch_size': 32,
            'eval_epoch': 100,
            'result_root': '/home/ubuntu/speech2vec/result/dsp_hw2/'
    }

    model_args = {
            'encode_dim': 10,
            'hidden_dim': 128,
            'nb_filters':32,
            'nb_conv': 3,
            'depth': (1,1),
            'dropout_keep_prob': 0.8
    }

    run_training(model_args, training_args)




