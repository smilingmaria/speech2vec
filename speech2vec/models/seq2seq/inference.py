import tensorflow as tf

def gaussian_inference( latent_dim, last_output, keep_prob ):
    
    hidden_dim = int( last_output.get_shape()[1] )

    last_output_drop = tf.nn.dropout( last_output, keep_prob )  

    # Weight matrics for Gaussian Variables
    W_mu = tf.get_variable("W_mu", shape = [ hidden_dim, latent_dim ] )
    b_mu = tf.get_variable("b_mu", shape = [ latent_dim ] ) 

    W_logvar = tf.get_variable("W_logvar", shape = [ hidden_dim, latent_dim ] )
    b_logvar = tf.get_variable("b_logvar", shape = [ latent_dim ] )
    
    # Mean and Variance prediction of gaussian 
    z_mean = tf.matmul( last_output_drop, W_mu ) + b_mu
    z_logvar = tf.matmul( last_output_drop, W_logvar ) + b_logvar

    # Sampling
    epsilon = tf.random_normal( tf.shape(z_logvar) )
    z = z_mean + tf.mul( tf.exp( 0.5 * z_logvar ), epsilon )

    return z, z_mean, z_logvar 
