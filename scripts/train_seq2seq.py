import pdb
import pickle

from seq2seq.models import Seq2seq
from keras import backend as K
import numpy as np
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

sample, timestep, feature = X.shape

reverse = False
batch_size = 32
nb_epoch = 1000
dropout = 0.1

hidden = 50

encoder_depth = 2
decoder_depth = 2

model_name = "batch_{0}_epoch_{1}_hidden_{2}_depth_{3},{4}_dropout_{5}".format(batch_size,nb_epoch,hidden,encoder_depth,decoder_depth,dropout) 

if reverse:
    model_name += "_reverse"

print "Compiling model..."
# Define Model

model = Seq2seq(batch_input_shape=(batch_size, timestep, feature), \
                hidden_dim=hidden, \
                output_length=timestep, \
                output_dim=feature, \
                depth=(encoder_depth,decoder_depth),
                dropout=dropout)

model.compile(loss='mse',optimizer='rmsprop', sample_weight_mode='temporal')

X = np.vstack([ X, X[: batch_size - X.shape[0] % batch_size] ] )
mask = loss_fnt_masks(X)

X_in = ( X[:,:,::-1] if reverse else X )

#history = model.fit( X, X, batch_size, nb_epoch = 1, sample_weight = mask)
#import pdb; pdb.set_trace()
for epoch in range( nb_epoch / 50 ): 
    print "Training epoch {}".format(50 * (epoch+1))
    history = model.fit( X_in, X, batch_size, nb_epoch = 50, sample_weight = mask)

    print "Saving model..."
    # Model architecture
    epoch_model_name = model_name + "_epoch_{}".format(50*(epoch+1))
    json_string = model.to_json()
    open(result_dir + epoch_model_name + '.json', 'w').write(json_string)

    # Model weights
    model.save_weights(result_dir + epoch_model_name + '_weights.h5')

    # Model training history
    open(result_dir + epoch_model_name + '.history','w').write(str(history.history))

    print "Saving reconstruction..."

# Model prediction
import pdb; pdb.set_trace()
"""
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
"""
