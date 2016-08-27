import tensorflow as tf

def fullyconnected2D( output_dim, input_tensor, keep_prob, name = None ):
    input_dim = int(input_tensor.get_shape()[1])
    with tf.variable_scope( name or "fullyconnected2D" ):
        W = tf.get_variable("W", shape = [input_dim, output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.zeros(shape=[output_dim])

        input_tensor_drop = tf.nn.dropout( input_tensor, keep_prob)

        output_tensor = tf.matmul( input_tensor_drop, W ) + b

        output_tensor_relu = tf.nn.relu( output_tensor )

    return output_tensor_relu




