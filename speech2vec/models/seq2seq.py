from collections import defaultdict

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from encoders import basic_encoder, bidirectional_encoder
from decoders import basic_decoder, attention_decoder

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

class Seq2seq(object):
    def __init__(self, batch_input_shape, cells, hidden_dim, depth, **kwargs):
        """ 
        Arguments:
            seq_shape: ( timestep, feature ) 
            cells: [ c1, ... ]
            depth: ( encoder_depth, decoder_depth )
            attention: bidirectional rnn as encoder and attention decoder
            peek: Boolean # If is not attention
        """
        # 'peek'(decoer), 'bidirectional'(encoder)
        self.model_options = defaultdict(bool, kwargs)
        
        # Extract info from constructor arguments
        self.batch_input_shape = batch_input_shape

        en_depth, de_depth = depth
      
        en_cell = get_cell( cells[0] )( hidden_dim ) 
        de_cell = get_cell( cells[1] )( hidden_dim ) 
        
        if self.model_options['bidirectional']:
            self.en_cell = [ rnn_cell.MultiRNNCell([ en_cell ] * en_depth), 
                    rnn_cell.MultiRNNCell([ en_cell ] * en_depth) ]
        else:
            self.en_cell = [ rnn_cell.MultiRNNCell([en_cell] * en_depth) ]
        
        self.de_cell = [ rnn_cell.MultiRNNCell([de_cell] * de_depth) ]
        

    def build_inputs(self):
        batch_size, timestep, feature = self.batch_input_shape
        self.x = tf.placeholder(tf.float32, shape=[ batch_size, timestep, feature])
        self.keep_prob = tf.placeholder(tf.float32)
    
    def build_encoder(self):
        bidirectional = self.model_options['bidirectional']
        
        if bidirectional:
            cell = self.en_cell[0]
            self.en_cell = [ cell ]
            self.code, _ = bidirectional_encoder( self.en_cell, self.x, self.keep_prob ) 
        else:
            self.code = basic_encoder( self.en_cell, self.x, self.keep_prob ) 

    def build_decoder(self):
        peek = self.model_options['peek']
       
        self.x_rec = basic_decoder( self.batch_input_shape, self.de_cell, self.code,  self.keep_prob, peek = peek ) 

    def build_loss(self):
        self.cost = tf.reduce_mean(tf.reduce_sum(tf.square(self.x_rec-self.x),[1,2]))

"""
    Same as Seq2seq, but with bidirectional rnn as encoder and attention_decoder
"""
class AttentionSeq2seq(Seq2seq):
    def __init__(self, batch_input_shape, cells, hidden_dim, depth, **kwargs):
        super(AttentionSeq2seq,self).__init__( batch_input_shape, cells, hidden_dim, depth, bidirectional = True, **kwargs)

    def build_encoder(self):
        self.code, self.annotation = bidirectional_encoder( self.en_cell, self.x, self.keep_prob ) 

    def build_decoder(self):
        self.x_rec = attention_decoder( self.batch_input_shape, self.de_cell, self.code, self.annotation, self.keep_prob ) 
        
