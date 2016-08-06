import pdb
import pickle

from keras.models import Model
from seq2seq.models import Seq2seq
from keras import backend as K

import numpy as np
import h5py

from speech2vec.tiling import build_annoy_tree, tile_reconstruction

from util import load_digits, save_to_csv
from util import loss_fnt_masks
from util import makedir


exp = 'dsp_hw2_one_digit_seq2seq'
fbank_type = 'fbank_delta'
result_dir = './result/' + exp + '/' + fbank_type + '/'
makedir(result_dir)

print "Loading dataset..."
_, X, y = load_digits()

sample, timestep, feature = X.shape

# Build Annoy Tree
ann = build_annoy_tree(X)

reverse = True
batch_size = 64
nb_epoch = 10000
dropout = 0.2

hidden = 100

encoder_depth = 2
decoder_depth = 2

model_name = "batch_{0}_epoch_{1}_hidden_{2}_depth_{3},{4}_dropout_{5}_optimizer_{6}"\
        .format(batch_size,nb_epoch,hidden,encoder_depth,decoder_depth,dropout,'nadam')

#model_name = 'debug'

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

def mean_squared_error(y_true, y_pred):
    if reverse:
        return K.mean(K.square(y_pred[:,:,::-1] - y_true), axis=-1)
    else:
        return K.mean(K.square(y_pred - y_true), axis=-1)

model.compile(loss=mean_squared_error,optimizer='nadam')

toadd = ( batch_size - X.shape[0] % batch_size ) % batch_size

X = np.vstack([ X, X[:toadd] ] )
#mask = loss_fnt_masks(X)

X_in = ( X[:,:,::-1] if reverse else X )

encoder = Model(input=model.layers[0].input, output=model.layers[2].output)

for epoch in range( 50, nb_epoch+1, 50 ):
    print "Training to epoch {}".format(epoch)

    history = model.fit( X_in, X, batch_size, nb_epoch = 50)

    print "Saving model..."
    # Model architecture
    epoch_model_name = model_name + "_epoch_{}".format(epoch)
    json_string = model.to_json()
    open(result_dir + epoch_model_name + '.json', 'w').write(json_string)

    # Model weights
    model.save_weights(result_dir + epoch_model_name + '_weights.h5', overwrite=True)

    # Model training history
    open(result_dir + epoch_model_name + '.history','w').write(str(history.history))

    rec_X = model.predict(X_in,  batch_size = batch_size)[:sample]
    code = encoder.predict(X_in, batch_size = batch_size)[:sample]

    # Find nearest neighbor with annoy tree
    tiled_X = tile_reconstruction(rec_X, ann)

    tiled_loss = ( X[:sample] - tiled_X ).mean()

    # Fbank numpy array
    h5f = h5py.File(result_dir + epoch_model_name + '.h5','w')
    h5f.create_dataset('rec_X',data=rec_X)
    h5f.create_dataset('tiled_X',data=tiled_X)
    h5f.create_dataset('code', data=code)
    h5f.create_dataset('y', data=y)
    h5f.close()
