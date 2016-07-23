import pdb
import pickle

from keras import backend as K
from speech2vec import variational_autoencoder
import numpy as np
import h5py
from util import load_dataset, save_to_csv
from util import loss_fnt_masks
from util import makedir

exp = 'vae_seq2seq'

dataset = './data/dev-clean/'
fbank_type = 'fbank_delta/'

result_dir = './result/' + exp + '/' + fbank_type + '/'
makedir(result_dir)

print "Loading dataset..."
#X = load_dataset(dataset+fbank_type)[:100]
X = np.random.random((100,20,5))
mask = loss_fnt_masks(X)

sample, timestep, feature = X.shape

batch_size = 16
nb_epoch = 2

rnn_type = 'LSTM'
hidden_dim = 50
encode_dim = 5
encoder_depth = 1
decoder_depth = 1
dropout = 0.25

model_name = "rnn_{0}_batch_{1}_epoch_{2}_hidden_{3}_encode_dim_{4}_depth_{5},{6}_dropout_{7}".\
format(rnn_type,batch_size,nb_epoch,hidden_dim,encode_dim,encoder_depth,decoder_depth,dropout) 

print "Compiling model..."
# Define Model

model,_,_ = variational_autoencoder( rnn_type = rnn_type,
				shape = ( batch_size, timestep, feature ),
				output_length = timestep,
				output_dim = feature,
				hidden_dim = hidden_dim,
				depth = ( encoder_depth, decoder_depth),
				encode_dim = encode_dim )

history = model.fit( X, X, batch_size, nb_epoch, sample_weight = mask)

print "Saving model..."
# Model architecture
json_string = model.to_json()
open(result_dir + model_name + '.json', 'w').write(json_string)

# Model weights
model.save_weights(result_dir + model_name + '_weights.h5')
# Model training history
open(result_dir+model_name+'.history','w').write(str(history.__dict__))
