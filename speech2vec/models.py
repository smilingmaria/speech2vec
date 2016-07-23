from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, Lambda, TimeDistributed, Dropout, Activation
from keras.layers import SimpleRNN, GRU, LSTM, onetomanySimpleRNN, onetomanyGRU, onetomanyLSTM
from keras import objectives
import numpy as np


def get_rnn_type(rnn_type):
	if rnn_type == 'SimpleRNN':
		return SimpleRNN, onetomanySimpleRNN	
	elif rnn_type == 'GRU':
		return GRU, onetomanyGRU
	elif rnn_type == 'LSTM':
		return LSTM, onetomanyLSTM
	else:
		raise Exception("Unknown rnn_type: {}".format(rnn_type))

def seq2seq_autoencoder(rnn_type, shape, output_length, output_dim, hidden_dim, encode_dim, depth=1, dropout=0.2,loss='mse',optimizer='rmsprop'):
        """
		Get rnn types and initialization
	"""
	timestep, feature = shape
	rnn, onetomany_rnn = get_rnn_type(rnn_type);

	if type(depth) not in [list, tuple]:
            depth = (depth, depth)
	enc_depth, dec_depth = depth	
	
	# Define encoder
	x = encoder_out = Input(shape=(timestep,feature))		
	encoder_out = Masking(mask_value=0.)(encoder_out) # Add masking
	encoder_out = TimeDistributed(Dense(hidden_dim))(encoder_out)	
	for i in range(1,enc_depth):
		encoder_out = rnn( hidden_dim, return_sequences=True )(encoder_out)		
		encoder_out = Dropout(dropout)(encoder_out)
	code = rnn( encode_dim ) (encoder_out)
	
	encoder = Model(input=x,output=code)	
	
	# Define decoder
	c = decoder_out = Input(shape=(encode_dim,))	
	decoder_out = onetomany_rnn(output_length=timestep, output_dim=hidden_dim)(decoder_out)
		
	for i in range(1,dec_depth):
		decoder_out = rnn( output_dim = hidden_dim, return_sequences =True)(decoder_out)
		decoder_out = Dropout(dropout)(decoder_output)		
	decoder_out = TimeDistributed(Dense(output_dim))(decoder_out)
	
	decoder = Model(input=c, output=decoder_out)	
	
	# Define Overall Model
	model = Sequential()
	model.add(encoder)
	model.add(decoder)	
	model.compile(loss=loss,optimizer=optimizer)	

	return model, encoder, decoder
	
def variational_autoencoder(rnn_type, shape, output_length, output_dim, hidden_dim, encode_dim, epsilon_std = 0.01, depth=1, dropout=0.2, loss = None, optimizer='rmsprop'):
	
	batch_size, timestep, feature = shape
	rnn, onetomany_rnn = get_rnn_type(rnn_type);

	if type(depth) not in [list, tuple]:
            depth = (depth, depth)
	enc_depth, dec_depth = depth	
	
	# Define encoder
	x = encoder_out = Input(shape=(timestep,feature))		
	encoder_out = Masking(mask_value=0.)(encoder_out) # Add masking
	encoder_out = TimeDistributed(Dense(hidden_dim))(encoder_out)	
	for i in range(1, enc_depth):
		encoder_out = rnn( hidden_dim, return_sequences=True )(encoder_out)		
		encoder_out = Dropout(dropout)(encoder_out)

	encoder_out = rnn( hidden_dim ) (encoder_out)

	z_mean = Dense(encode_dim)(encoder_out)
	z_log_std = Dense(encode_dim)(encoder_out)

	def sampling(args):
		z_mean, z_log_std = args
    		epsilon = K.random_normal(shape=(batch_size, encode_dim),
                              mean=0., std=epsilon_std)
    		return z_mean + K.exp(z_log_std) * epsilon	
	
	z = Lambda(sampling, output_shape=(encode_dim,))([z_mean, z_log_std])

	encoder = Model(input=x,output=z_mean)	
	
	# Define decoder
	c = decoder_out = Input(shape=(encode_dim,))	
	decoder_out = onetomany_rnn(output_length=timestep, output_dim=hidden_dim)(decoder_out)
		
	for i in range(1,dec_depth):
		decoder_out = rnn( output_dim = hidden_dim, return_sequences =True)(decoder_out)
		decoder_out = Dropout(dropout)(decoder_output)		

	decoder_out = TimeDistributed(Dense(output_dim))(decoder_out)
	
	decoder = Model(input=c, output=decoder_out)	

	def vae_loss(x, x_decoded_mean):
    		xent_loss = objectives.mean_squared_error(x, x_decoded_mean)
    		kl_loss = - 0.5 * K.mean(1 + z_log_std - K.square(z_mean) - K.exp(z_log_std), axis=-1)
    		return xent_loss + kl_loss
	
	# Define Overall Model
	model = Sequential()
	model.add(encoder)
	model.add(decoder)	
	model.compile(loss=vae_loss,optimizer=optimizer)	
	
	return model, encoder, decoder
	
if __name__ == "__main__":
	#s2s = seq2seq_autoencoder('GRU',(2,3),4,5,6,7)
	vae = variational_autoencoder('SimpleRNN',(2,3,4),5,6,7,8)
