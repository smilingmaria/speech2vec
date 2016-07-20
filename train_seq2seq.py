import pdb
import pickle

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Masking, RepeatVector, Dropout
from keras.layers import recurrent,LSTM
from keras.layers.wrappers import TimeDistributed

import h5py
from util import load_dataset, save_to_csv
from util import loss_fnt_masks
from util import makedir

exp = 'simpleseq2seq'
result_dir = './result/' + exp + '/'

makedir(result_dir)

dataset = 'dev'
fbank_type = 'fbank/'

print "Loading dataset..."
X = load_dataset(dataset,fbank_type)
mask = loss_fnt_masks(X)

sample, timestep, feature = X.shape

batch_size = 32
nb_epoch = 1
dropout = 0.25

hidden = 10

print "Compiling model..."
# Define Model
x = Input(shape=(timestep,feature))

masked_x = Masking(mask_value=0.)(x)
#x = Masking(mask_value=0.,input_shape=(timestep,feature))

#enc1 = LSTM(hidden)(x)
enc1 = LSTM(hidden)(masked_x)

code = enc1

dropped_code = Dropout(dropout)(code)

repeated_dropped_code = RepeatVector(timestep)(dropped_code)

dec1 = LSTM(hidden, return_sequences=True)(repeated_dropped_code)

rec_x = TimeDistributed(Dense(75))(dec1)

model = Model(input = [x], output=[rec_x])
encoder = Model(input=[x],output=[code])

model.compile(loss='mse', optimizer='rmsprop',sample_weight_mode='temporal')
history = model.fit( X, X, 
            batch_size,
            nb_epoch,
            sample_weight = mask
            )

print "Saving model..."
# Model architecture
json_string = model.to_json()
open(result_dir + exp + '.json', 'w').write(json_string)

# Model weights
model.save_weights(result_dir + exp + '_weights.h5')

# Model training history
pickle.dump(history,open(result_dir + exp + '.history','w'))

print "Saving reconstruction..."
# Model prediction
rec_X = model.predict(X)
code = encoder.predict(X)
# CSV
csv_dir = result_dir + 'csv/'
makedir(csv_dir)
save_to_csv(Y,csv_dir)
# Fbank numpy array
h5f = h5py.File(result_dir + exp + '_result.h5','w')
h5f.create_dataset('rec_X',data=rec_X)
h5f.create_dataset('code', data=code)
h5f.close()
