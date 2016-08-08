import time
import numpy as np
import tensorflow as tf

import config
from speech2vec.datasets import dsp_hw2
from speech2vec.models import Seq2seq

# Load data
dataset = dsp_hw2()

X = dataset.fbank_delta
y = dataset.labels

# Learning Parameters
sample, timestep, feature = X.shape

cells = ['GRUCell'] * 2

batch_size = 32
hidden_dim = 128
depth = (1,1)
keep_prob = 0.8
peek = False

batch_input_shape = ( batch_size, timestep, feature )

# Build model
model = Seq2seq( batch_input_shape, cells, hidden_dim, depth, bidirectional=True)
model.build_graph()

# Training
nb_epochs = 1000

# GPU memory alllocation
saver = tf.train.Saver()

min_loss = 1.
epoch_losses = []

with tf.Session() as sess:
    
    tf.initialize_all_variables().run()

    for epoch in range(1, nb_epochs+1, 1):
        epoch_loss = 0.
        batch_counter = 0
        for x in dataset.next_batch(norm = False, batch_size = batch_size):
            feed = { model.x : x, model.keep_prob: keep_prob }
            train_loss, _, = sess.run( [ model.cost, model.optimizer ], feed_dict = feed )
            epoch_loss += train_loss
            batch_counter += 1
        
        
        epoch_loss /= batch_counter
        epoch_losses.append(epoch_loss)
        if epoch_loss < min_loss: 
            min_loss = epoch_loss
            save_path = saver.save(sess, "../result/attention.ckpt")
            print("Model saved in file: %s" % save_path)
        print "Epoch {}, loss {}, min_loss {}".format( epoch, epoch_loss, min_loss) 
