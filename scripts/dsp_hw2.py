import sys
import time

import numpy as np
import tensorflow as tf

import config
from speech2vec.datasets import dsp_hw2 as load_dataset
from speech2vec.models import Seq2seqAutoencoder
from speech2vec.evaluation import save_h5, save_to_csv
from speech2vec.utils import makedir
# Load data
dataset = load_dataset()

result_dir = '../result/' + load_dataset.__name__ + '/'

X = dataset.X
y = dataset.y

# Learning Parameters
sample, timestep, feature = X.shape

cells = ['GRUCell'] * 2

nb_epochs = 5000
batch_size = 128
hidden_dim = 512
depth = (1,1)
keep_prob = 0.8
peek = False
bidirectional = True

batch_input_shape = ( batch_size, timestep, feature )

# Build model
model = Seq2seqAutoencoder( batch_input_shape,\
                            cells,\
                            hidden_dim,\
                            depth,\
                            keep_prob,\
                            peek = peek,\
                            bidirectional=bidirectional)

model.build_graph()

model_name = model.name

save_path = result_dir + model_name + '.ckpt'

# GPU memory alllocation
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

saver = tf.train.Saver()

min_loss = sys.float_info.max
with tf.Session(config=config) as sess:
    

    tf.initialize_all_variables().run()

    for epoch in range(1, nb_epochs+1, 1):
        
        epoch_loss = model.train_one_epoch( sess, dataset.next_batch(batch_size=batch_size,shuffle=True) )
        if epoch_loss < min_loss: 
            min_loss = epoch_loss
            model.save(sess, saver, save_path)
        
        print "Epoch {}, loss {}, min_loss {}".format( epoch, epoch_loss, min_loss)
    
    # Load the model with the lowest reconstruction error
    print "Min loss",min_loss
    model.load(sess, saver, save_path)
    
    X_rec = model.reconstruct(sess,dataset.next_batch(batch_size=batch_size, shuffle = False))
    code  = model.encode(sess,dataset.next_batch(batch_size=batch_size, shuffle = False)) 
  
    # Define file save paths
    minloss_modelname = model_name + '_minloss_{}'.format(min_loss)
    save_dir = result_dir + minloss_modelname + '/'
    makedir(save_dir)
    
    # Save model
    save_path = save_dir + minloss_modelname + '.ckpt'
    model.save(sess, saver, save_path) 
    
    # Save Reconstruction * code
    h5_path   = save_dir + minloss_modelname + '.h5'
    feat, phase = dataset.split_X(X_rec)
    save_h5( h5_path, feat, phase, code)

    # Save to csvs for matlab
    feature_path = save_dir + dataset.data_type + '/'
    save_to_csv( feat, feature_path )

    yphase_path = save_dir + 'yphase/'
    save_to_csv( phase, yphase_path ) 
