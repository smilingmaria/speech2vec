import sys
import time
import numpy as np
import tensorflow as tf

import config
from speech2vec.datasets import dsp_hw2 as load_dataset
from speech2vec.models import Seq2seqAutoencoder

# Load data
dataset = load_dataset()

result_dir = '../result/' + load_dataset.__name__ + '/'

X = dataset.X
y = dataset.y

# Learning Parameters
sample, timestep, feature = X.shape

cells = ['GRUCell'] * 2

nb_epochs = 1
batch_size = 64
hidden_dim = 128
depth = (1,1)
keep_prob = 0.8
peek = False
bidirectional = False

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

# GPU memory alllocation
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

saver = tf.train.Saver()

min_loss = sys.float_info.max

with tf.Session(config=config) as sess:
    
    tf.initialize_all_variables().run()

    for epoch in range(1, nb_epochs+1, 1):
        
        epoch_loss = model.train_one_epoch( sess, dataset.next_batch(batch_size=batch_size) )
        
        if epoch_loss < min_loss: 
            min_loss = epoch_loss
            save_path = result_dir + model_name + '.ckpt'
            model.save(sess, saver, save_path)
        
        print "Epoch {}, loss {}, min_loss {}".format( epoch, epoch_loss, min_loss)
        print "Test loss", model.test(sess, dataset.next_batch(batch_size=batch_size))
    load_path = save_path
    model.load(sess, saver, load_path)
    
    X_rec = model.reconstruct(sess,dataset.next_batch(batch_size=batch_size, shuffle = False))
    code  = model.encode(sess,dataset.next_batch(batch_size=batch_size, shuffle = False)) 
    
    import pdb; pdb.set_trace()
    
    X_rec = X_rec[:sample]
    code = code[:sample]
    
    print "QQ"

