import pdb
import pickle

from seq2seq.models import Seq2seq
from keras import backend as K

import h5py
from util import load_dataset, save_to_csv
from util import loss_fnt_masks
from util import makedir

exp = 'seq2seq'

dataset = './data/dev-clean/'
fbank_type = 'fbank_delta/'

result_dir = './result/' + exp + '/' + fbank_type + '/'
makedir(result_dir)

print "Loading dataset..."
X = load_dataset(dataset+fbank_type)
mask = loss_fnt_masks(X)

sample, timestep, feature = X.shape

batch_size = 32
nb_epoch = 20
dropout = 0.25

hidden = 20
encoder_depth = 2
decoder_depth = 2

model_name = "batch_{0}_epoch_{1}_hidden_{2}_depth_{3},{4}".format(batch_size,nb_epoch,hidden,encoder_depth,decoder_depth) 

print "Compiling model..."
# Define Model

model = Seq2seq(batch_input_shape=(batch_size, timestep, feature), \
                hidden_dim=hidden, \
                output_length=timestep, \
                output_dim=feature, \
                depth=(encoder_depth,decoder_depth))
model.compile(loss='mse',optimizer='rmsprop', sample_weight_mode='temporal')

history = model.fit( X, X, batch_size, nb_epoch, sample_weight = mask)

print "Saving model..."
# Model architecture
json_string = model.to_json()
open(result_dir + model_name + '.json', 'w').write(json_string)

# Model weights
model.save_weights(result_dir + model_name + '_weights.h5')
# Model training history
open(result_dir+model_name+'.history','w').write(str(history.__dict__))

print "Saving reconstruction..."
# Model prediction
encoder = K.function([model.layers[0].input,K.learning_phase()], model.layers[ 2 * encoder_depth - 1].output)

rec_X = model.predict(X)
code = encoder([X,False])
# CSV
#csv_dir = result_dir + 'csv/' + model_name + '/'
#makedir(csv_dir)
#save_to_csv(rec_X,csv_dir)

# Fbank numpy array
h5f = h5py.File(result_dir + model_name + '_result.h5','w')
h5f.create_dataset('rec_X',data=rec_X)
h5f.create_dataset('code', data=code)
h5f.close()
