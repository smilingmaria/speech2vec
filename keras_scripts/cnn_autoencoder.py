import os
import sys

import h5py
import numpy as np

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras import objectives


class CNNAutoencoder(object):
    def __init__(self, batch_input_shape, hidden_dim, latent_dim, nb_filters, nb_conv, epsilon_std = 0.01, **kwargs ):

    def name(self):
        pass

    def build_graph(self):
        pass

    def build_inputs(self):
        pass

    def build_encoder(self):
        pass

    def build_code(self):




# train the VAE on MNIST digits
def conv_vae( batch_input_shape, latent_dim=2, intermediate_dim=128, nb_filters=32, nb_conv=3, epsilon_std = 0.01):
    batch_size, img_chns, img_rows, img_cols = batch_input_shape
    # Input
    x = Input(batch_shape=batch_input_shape)

    # Encoder
    c = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', activation='relu')(x)
    f = Flatten()(c)

    # Sampling
    h = Dense(intermediate_dim, activation='relu')(f)

    z_mean    = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., std=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])


    # Decoder
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_f = Dense(nb_filters*img_rows*img_cols, activation='relu')
    decoder_c = Reshape((nb_filters, img_rows, img_cols))
    decoder_mean = Deconvolution2D(img_chns, nb_conv, nb_conv,
                                   (batch_size, img_chns,img_rows, img_cols),
                                   border_mode='same')

    h_decoded = decoder_h(z)
    f_decoded = decoder_f(h_decoded)
    c_decoded = decoder_c(f_decoded)
    x_decoded_mean = decoder_mean(c_decoded)

    # Define loss
    def vae_loss(x, x_decoded_mean):
        # NOTE: binary_crossentropy expects a batch_size by dim for x and x_decoded_mean, so we MUST flatten these!
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = objectives.mean_squared_error(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss +  kl_loss

    # Compile Model
    vae = Model(x, x_decoded_mean)
    vae.compile(optimizer='rmsprop', loss=vae_loss)
    vae.summary()

    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)

    return vae, encoder

def run_training( data_h5, data_type, training_args, model_args, result_dir ):
    # Read params
    nb_epochs  = training_args.get('nb_epochs',10)
    batch_size = training_args.get('batch_size',16)

    latent_dim = model_args.get('latent_dim',2)
    intermediate_dim = model_args.get('intermediate_dim',128)
    nb_filters = model_args.get('nb_filters',32)
    nb_conv = model_args.get('nb_cols',3)

    model_name = get_model_name( data_type, training_args, model_args)

    save_dir = os.path.join(result_dir,model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Preprocess data for model input
    with h5py.File(data_h5,'r') as data_handle:
        feature = data_handle[ data_type ][:]

    feature = feature[:,None,:,:] # Add one more dimension
    nb_samples, nb_chns, nb_rows, nb_cols = feature.shape

    batch_input_shape = ( batch_size, nb_chns, nb_rows, nb_cols )
    X = np.concatenate([ feature, feature[:batch_size - nb_samples % batch_size]])

    # Load model
    vae, encoder = conv_vae( batch_input_shape = batch_input_shape,
                             latent_dim = latent_dim,
                             intermediate_dim = intermediate_dim,
                             nb_filters = nb_filters,
                             nb_conv = nb_conv )

    # Start training
    for epoch in range(1, nb_epochs, 1):
        print("Epoch {}:".format(epoch))
        vae.fit(X, X,
                shuffle=True,
                nb_epoch=1,
                batch_size=batch_size)


        epoch_save_path = os.path.join(save_dir, 'epoch{}.h5'.format(epoch))
        print("Saving to {}".format(epoch_save_path))
        recX = vae.predict(X, batch_size=batch_size)
        recX = recX[:nb_samples]
        code = encoder.predict(X, batch_size=batch_size)
        code = code[:nb_samples]

        with h5py.File(epoch_save_path,'w') as handle:
            handle.create_dataset('code',data=code)
            handle.create_dataset('recX',data=recX)

def get_model_name( data_type, training_args, model_args ):
    arg_list = list(training_args.items()) + list(model_args.items())

    name = data_type + "_CNN"
    for k,v in arg_list:
        if name != data_type + "CNN":
            name += "_"
        name += k + str(v)

    return name

if __name__ == "__main__":
    # Data save & load locations
    dataset_root = '/home/ubuntu/speech2vec/raw_data/dsp_hw2/'
    data_h5 = os.path.join(dataset_root,'data.h5')
    result_dir = '/home/ubuntu/speech2vec/result/dsp_hw2'

    data_type = 'fbank_delta'

    training_args = {
            'nb_epochs': 1000,
            'batch_size': 16
            }

    model_args = {
            'latent_dim': 2,
            'intermediate_dim': 128,
            'nb_filters':32,
            'nb_conv': 3
    }


    run_training( data_h5, data_type, training_args, model_args, result_dir )




