import time
import numpy as np
import tensorflow as tf

import config
from speech2vec.datasets import wav
from speech2vec.models import Seq2seq

# Load data
dataset = wav()

X = dataset.X
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
print "Building model..."
model = Seq2seq( batch_input_shape,\
                            cells,\
                            hidden_dim,\
                            depth)
model.build_graph()

# Training
nb_epochs = 1000

# GPU memory alllocation
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6

saver = tf.train.Saver()

min_loss = 1.
epoch_losses = []
print "Start training"
with tf.Session(config=config) as sess:
    
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
            save_path = saver.save(sess, "../result/wav/seq2seq_gru.ckpt")
            print("Model saved in file: %s" % save_path)
        print "Epoch {}, loss {}, min_loss {}".format( epoch, epoch_loss, min_loss) 
