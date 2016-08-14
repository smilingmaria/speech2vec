import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.ops import seq2seq

# Assume input "x" is a 3D matrix of ( sample, timestep, feature )

def basic_encoder( cells, x, keep_prob ):
    assert len(cells) == 1, "One cell needed!"
    en_cell = cells[0]
    
    encoder_inputs = tf.unpack(tf.transpose(x, perm=[1,0,2]))

    encoder_inputs_dropout = [ tf.nn.dropout( inp, keep_prob ) \
            for inp in encoder_inputs ] 
    
    encoder_outputs, encoder_state = \
            tf.nn.rnn(en_cell, encoder_inputs_dropout, dtype=tf.float32)
    
    code = tf.nn.relu(encoder_outputs[-1])

    return code

def bidirectional_encoder( cells, x, keep_prob ):
    assert len(cells) == 2, "Two cells needed for bidirectional_rnn!"
    cell_fw, cell_bw = cells

    encoder_inputs = tf.unpack(tf.transpose(x, perm=[1,0,2]))
    
    encoder_inputs_dropout = [ tf.nn.dropout( inp, keep_prob ) \
            for inp in encoder_inputs ] 
    
    concat_outputs, fw_state, bw_state  = \
            rnn.bidirectional_rnn(cell_fw, cell_bw, encoder_inputs_dropout, dtype=tf.float32)

    def split_then_add( tens ):
        split0, split1 = tf.split( 1, 2, tens)
        return split0 + split1

    outputs = map(split_then_add, concat_outputs)

    code = tf.nn.relu(outputs[-1])

    return code, outputs

