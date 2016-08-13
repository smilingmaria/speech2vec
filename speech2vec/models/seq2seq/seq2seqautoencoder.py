from collections import defaultdict

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from .inference import gaussian_inference 
from .encoders import basic_encoder, bidirectional_encoder
from .decoders import basic_decoder, attention_decoder

def get_cell(cell_type):
    if isinstance(cell_type,str):
        cell_dict = { 
            'BasicRNNCell': rnn_cell.BasicRNNCell, 
            'BasicLSTMCell': rnn_cell.BasicLSTMCell,
            'GRUCell': rnn_cell.GRUCell,
            'LSTMCell': rnn_cell.LSTMCell 
             }
        return cell_dict[ cell_type ] 
    else:
        return cell_type

class Seq2seqAutoencoder(object):
    def __init__(self, batch_input_shape, cells, hidden_dim, depth, dropout_keep_prob, **kwargs):
        """ 
        Arguments:
            seq_shape: ( timestep, feature ) 
            cells: [ c1, ... ]
            depth: ( encoder_depth, decoder_depth )
            attention: bidirectional rnn as encoder and attention decoder
            peek: Boolean # If is not attention
        """
        # 'peek'(decoder), 'bidirectional'(encoder)
        self.model_options = defaultdict(bool, kwargs)
        
        # Extract info from constructor arguments
        self.batch_input_shape = batch_input_shape

        en_depth, de_depth = depth
      
        en_cell = get_cell( cells[0] )( hidden_dim ) 
        de_cell = get_cell( cells[1] )( hidden_dim ) 
        
        if self.model_options['bidirectional']:
            self.en_cell = [ rnn_cell.MultiRNNCell([ en_cell ] * en_depth, state_is_tuple = True), 
                    rnn_cell.MultiRNNCell([ en_cell ] * en_depth, state_is_tuple = True) ]
        else:
            self.en_cell = [ rnn_cell.MultiRNNCell([en_cell] * en_depth, state_is_tuple = True) ]
        
        self.de_cell = [ rnn_cell.MultiRNNCell([de_cell] * de_depth, state_is_tuple=True) ]
      
        self.dropout_keep_prob = dropout_keep_prob 
        
        # Define model name according to structure
        namestring = ""
        namestring += self.__class__.__name__
        namestring += '_' + '_'.join([ t[0] + str(t[1]) for t in zip(cells,depth) ])
        namestring += '_hidden_' + str(hidden_dim)
        if self.model_options['peek']:
            namestring += '_peek'
        if self.model_options['bidirectional']:
            namestring += '_bidirectional'
        self._name = namestring
    
    @property
    def name(self): 
        return self._name
    
    ###################
    #   Model Build   #
    ###################
    def build_graph(self):
        self.build_inputs()
        self.build_encoder()
        self.build_decoder()
        self.build_loss()
        self.build_optimizer()

    def build_inputs(self):
        batch_size, timestep, feature = self.batch_input_shape
        self.x = tf.placeholder(tf.float32, shape=[batch_size, timestep, feature])
        self.keep_prob = tf.placeholder(tf.float32)
    
    def build_encoder(self):
        bidirectional = self.model_options['bidirectional']
        
        if bidirectional:
            self.code, _ = bidirectional_encoder( self.en_cell, self.x, self.keep_prob ) 
        else:
            self.code = basic_encoder( self.en_cell, self.x, self.keep_prob ) 

    def build_decoder(self):
        peek = self.model_options['peek']
       
        self.x_rec = basic_decoder( self.batch_input_shape, self.de_cell, self.code,  self.keep_prob, peek = peek ) 

    def build_loss(self):
        self.cost = tf.reduce_mean(tf.square(self.x_rec-self.x))

    def build_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.cost)

    ##################
    #    Training    #
    ##################
   
    # Assumes generator iterates through one epoch of data
    def train_one_epoch(self, sess, generator):
        epoch_loss = 0.
        for batch_count, (x, y) in enumerate( generator ):
            
            feed = { self.x: x, self.keep_prob: self.dropout_keep_prob }
            
            loss, _ = sess.run([ self.cost, self.optimizer], feed_dict = feed)
            
            epoch_loss += loss
        
        epoch_loss /= ( batch_count + 1 ) 
        
        return epoch_loss

    # Iterates through generator data
    def test(self, sess, generator ):
        epoch_loss = 0.
        for batch_count, (x, y) in enumerate( generator ):
            
            feed = { self.x: x, self.keep_prob: 1. }
            
            loss = sess.run( self.cost, feed_dict = feed)
            
            epoch_loss += loss
        
        epoch_loss /= ( batch_count + 1 ) 
        return epoch_loss
    
    # Reconstruct method
    def reconstruct(self, sess, generator ):
        
        X_rec = []

        for batch_count, (x, y) in enumerate( generator ):
            feed = { self.x: x, self.keep_prob: 1. }
            x_rec = sess.run( self.x_rec, feed_dict = feed)
            X_rec.append(x_rec)
    
        X_rec = np.vstack(X_rec)

        return X_rec
    
    # Encode method
    def encode(self, sess, generator ):
        Code = []

        for batch_count, (x, y) in enumerate( generator ):
            feed = { self.x: x, self.keep_prob: 1. }
            code = sess.run( self.code, feed_dict = feed)
            Code.append(code)
    
        Code = np.vstack(Code)
        return Code
    
    ##################
    #    Utilities   #
    ################## 
    def save(self, sess, saver, save_path):
        save_path = saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    def load(self, sess, saver, load_path):
        saver.restore(sess, load_path)
        print("Model restored: %s" % load_path)

class VariationalSeq2seqAutoencoder(Seq2seqAutoencoder):
    def __init__(self, batch_input_shape, cells, hidden_dim, latent_dim, depth, dropout_keep_prob, **kwargs):
        super(VariationalSeq2seqAutoencoder, self).__init__(batch_input_shape, cells, hidden_dim, depth, dropout_keep_prob, **kwargs)
        self.hidden_dim = hidden_dim 
        self.latent_dim = latent_dim
        self._name += '_latent_' + str(latent_dim) 
    
    def build_graph(self):
        self.build_inputs()
        self.build_encoder()
        self.build_inference()
        self.build_decoder()
        self.build_loss()
        self.build_optimizer()

    def build_inference(self):
        self.encoder_lastoutput = self.code

        self.z, self.z_mean, self.z_logvar = \
                gaussian_inference( self.latent_dim, self.encoder_lastoutput, self.keep_prob )   
        # Work around for encode function 
        self.code = self.z_mean
   
    
    def build_decoder(self):
        peek = self.model_options['peek']
      

        W_de_init = tf.get_variable("W_de_init", shape = [self.latent_dim, self.hidden_dim],\
                                    initializer=tf.contrib.layers.xavier_initializer() )

        b_de_init = tf.Variable( tf.zeros([ self.hidden_dim ] ) ) 

        decoder_init = tf.matmul( tf.nn.dropout( self.z, self.keep_prob ), W_de_init ) + b_de_init 
        self.x_rec = basic_decoder( self.batch_input_shape, self.de_cell, decoder_init,  self.keep_prob, peek = peek ) 
    
    def build_loss(self):
        self.latent_cost = tf.reduce_mean( 
                - 0.5 * tf.reduce_sum(1 + self.z_logvar - tf.square(self.z_mean) - tf.exp( self.z_logvar ), reduction_indices = [1] )
                )
        self.rec_cost = tf.reduce_mean( 
                tf.reduce_sum(tf.square(self.x - self.x_rec), reduction_indices=[1,2])
                )
        self.cost = self.latent_cost + self.rec_cost

    ##################
    #    Training    #
    ##################
    
    def train_one_epoch(self, sess, generator):
        epoch_loss = 0.
        latent_loss = 0.
        rec_loss = 0.
        for batch_count, (x, y) in enumerate( generator ):
            
            feed = { self.x: x, self.keep_prob: self.dropout_keep_prob }
            fetches = [ self.cost, self.latent_cost, self.rec_cost, self.optimizer ] 
            
            loss, lat_cost, rec_cost, _ = sess.run( fetches, feed_dict = feed)
            
            epoch_loss += loss
            latent_loss += lat_cost
            rec_loss += rec_cost
        
        epoch_loss /= ( batch_count + 1 ) 
        latent_loss /= ( batch_count + 1 )
        rec_loss /= ( batch_count + 1 ) 
        
        return [ epoch_loss, latent_loss, rec_loss ]

    # Iterates through generator data
    def test(self, sess, generator):
        epoch_loss = 0.
        latent_loss = 0.
        rec_loss = 0.
        for batch_count, (x, y) in enumerate( generator ):
            
            feed = { self.x: x, self.keep_prob: self.dropout_keep_prob }
            fetches = [ self.cost, self.latent_cost, self.rec_cost ] 
            
            loss, lat_cost, rec_cost = sess.run( fetches, feed_dict = feed)
            
            epoch_loss += loss
            latent_loss += lat_cost
            rec_loss += rec_cost
        
        epoch_loss /= ( batch_count + 1 ) 
        latent_loss /= ( batch_count + 1 )
        rec_loss /= ( batch_count + 1 ) 
        
        return [ epoch_loss, latent_loss, rec_loss ]
    
    # Reconstruct method
    def reconstruct(self, sess, generator ):
        
        X_rec = []

        for batch_count, (x, y) in enumerate( generator ):
            feed = { self.x: x, self.keep_prob: 1. }
            x_rec = sess.run( self.x_rec, feed_dict = feed)
            X_rec.append(x_rec)
    
        X_rec = np.vstack(X_rec)

        return X_rec
    
    # Encode method
    def encode(self, sess, generator ):
        Code = []

        for batch_count, (x, y) in enumerate( generator ):
            feed = { self.x: x, self.keep_prob: 1. }
            code = sess.run( self.z_mean, feed_dict = feed)
            Code.append(code)
    
        Code = np.vstack(Code)
        return Code
   
    def generate(self, generator):
        pass
"""
    Same as Seq2seq, but with bidirectional rnn as encoder and attention_decoder
"""
class AttentionSeq2seqAutoencoder(Seq2seqAutoencoder):
    def __init__(self, batch_input_shape, cells, hidden_dim, depth, **kwargs):
        # 'peek'(decoder), 'bidirectional'(encoder)
        kwargs['bidirectional'] = True
        super(AttentionSeq2seqAutoencoder, self).__init__(batch_input_shape, cells, hidden_dim, depth, **kwargs)

    def build_encoder(self):
        # Recieves annoation from bidirectional encode
        self.code, self.annotation = bidirectional_encoder( self.en_cell, self.x, self.keep_prob ) 
    def build_decoder(self):
        self.x_rec = attention_decoder( self.batch_input_shape, self.de_cell, self.code, self.annotation, self.keep_prob ) 
