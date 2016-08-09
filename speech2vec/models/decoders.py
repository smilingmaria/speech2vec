import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.ops import seq2seq

from init import xavier_init

# Assume input "x" is a 3D matrix of ( sample, timestep, feature )
# Named arguments are mostly tensors
# Keyword arguments contain 'real numbers, such as timestep, hidden_dim'
# Basic decoder,  with peek option
def basic_decoder( batch_input_shape, cells, code,  keep_prob, **kwargs ):
    # Recieve arguments
    batch_size, timestep, feature = batch_input_shape
    peek = kwargs['peek']

    assert len(cells) == 1, "One cell needed!"
    de_cell = cells[0]
    
    # Start building graph
    hidden_dim = de_cell.output_size 

    code_dropout = tf.nn.dropout(code, keep_prob)
   
    code_dim = int(code_dropout.get_shape()[1])

    decoder_inputs = [ code_dropout ] + [ tf.placeholder(tf.float32, shape=[ batch_size, code_dim ] ) for i in range(timestep-1) ]
    
    def loop(prev, i):
        if peek:
            return prev + code_dropout # Output as input
        else:
            return prev
   
    decoder_outputs, decoder_state = seq2seq.rnn_decoder( decoder_inputs, de_cell.zero_state(batch_size,tf.float32), de_cell, loop_function = loop )
   
    W_out = tf.Variable( xavier_init(hidden_dim, feature) )
    b_out = tf.Variable( tf.zeros([ feature ] ) )

    unpacked_reconstruction = [ tf.matmul( tf.nn.dropout( out, keep_prob ), W_out ) for out in decoder_outputs ]

    recX = tf.nn.relu( tf.transpose(tf.pack(unpacked_reconstruction), perm=[1, 0, 2]) )

    return recX

def attention_decoder( batch_input_shape, cells, code, annotation, keep_prob, **kwargs ):
    # Recieve arguments
    batch_size, timestep, feature = batch_input_shape

    assert len(cells) == 1, "One cell needed!"
    de_cell = cells[0]
    
    hidden_dim = de_cell.output_size

    # Start building graph
    code_dropout = tf.nn.dropout(code, keep_prob)
    
    code_dim = int( code_dropout.get_shape()[1] ) 
    decoder_inputs = [ code_dropout ] + \
            [ tf.placeholder(tf.float32, shape=[ batch_size, code_dim ]  ) for i in range( timestep -1 ) ]

    def loop(prev, i):
            return prev # Output as input
    
    packed_annotation = tf.transpose(tf.pack(annotation), perm=[1,0,2])
   
    decoder_outputs, decoder_state = seq2seq.attention_decoder( decoder_inputs, de_cell.zero_state(batch_size,tf.float32), packed_annotation ,de_cell, loop_function = loop )
   
    W_out = tf.Variable( xavier_init(hidden_dim, feature) )
    b_out = tf.Variable( tf.zeros([ feature ] ) )

    unpacked_reconstruction = [ tf.matmul( tf.nn.dropout( out, keep_prob ), W_out ) for out in decoder_outputs ]

    recX = tf.nn.relu( tf.transpose(tf.pack(unpacked_reconstruction), perm=[1, 0, 2]) )

    return recX