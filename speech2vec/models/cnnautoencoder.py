import logging
import os
import sys

from annoy import AnnoyIndex
import h5py
import numpy as np

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D
from keras.models import Model, load_model
from keras import backend as K
from keras import objectives, optimizers

sys.path.insert(0, os.path.abspath('../../'))

from speech2vec.evaluation import eval_code_pr

class CNNAutoencoder(object):
    def __init__(self,
                 data_reader,
                 hidden_dim,
                 nb_filters,
                 nb_conv,
                 encode_dim,
                 depth,
                 dropout_keep_prob):

        self.data_reader = data_reader

        batch_size = self.data_reader.batch_size
        nb_rows, nb_cols = self.data_reader.shape[1:]
        self.batch_input_shape = (batch_size, nb_rows, nb_cols)

        self.hidden_dim = hidden_dim
        self.nb_filters = nb_filters
        self.nb_conv = nb_conv
        self.encode_dim = encode_dim

        print("Depth & Dropout are currently unused in CNN Autoencoders")
        logging.debug("Depth & Dropout are currently unused in CNN Autoencoders")

        self.depth = depth
        self.dropout_keep_prob = dropout_keep_prob

    @property
    def name(self):
        batch_size, nb_rows, nb_cols = self.batch_input_shape
        namestring = self.__class__.__name__
        namestring += '_batch{}_hidden{}_encode{}_filter{}_conv{}'\
                .format(batch_size, self.hidden_dim, self.encode_dim, self.nb_filters, self.nb_conv)
        return namestring

    def summary(self):
        self.model.summary()

    ######################
    #     Build Graph    #
    ######################
    def build_graph(self):
        self.build_inputs()
        self.build_encoder()
        self.build_code()
        self.build_decoder()
        self.build_output()
        self.build_loss()
        self.build_optimizer()

        self.build_models()

    def build_inputs(self):
        batch_size, nb_rows, nb_cols = self.batch_input_shape
        input_shape = (nb_rows, nb_cols)
        self.x = Input(shape=(nb_rows,nb_cols))

    def build_encoder(self):
        batch_size, nb_rows, nb_cols = self.batch_input_shape
        target_shape = ( 1, nb_rows, nb_cols )
        im_shape_x = Reshape( target_shape )(self.x)

        conv_output = Convolution2D( self.nb_filters,
                                     self.nb_conv,
                                     self.nb_conv,
                                     border_mode='same',
                                     activation='relu')(im_shape_x)
        self.encoder_output = Flatten()(conv_output)
        #flattened_conv_output = Flatten()(conv_output)

        #self.encoder_output = Dense(self.hidden_dim, activation='relu')(flattened_conv_output)

    def build_code(self):
        self.code = Dense(self.encode_dim, activation='relu')(self.encoder_output)

    def build_decoder(self):

        batch_size, nb_rows, nb_cols = self.batch_input_shape
        nb_chns = 1

        # Define layers
        fc_layer_1 = Dense(self.hidden_dim, activation='relu')
        fc_layer_2 = Dense(self.nb_filters * nb_rows * nb_cols, activation='relu')

        reshape_fc = Reshape( ( self.nb_filters, nb_rows, nb_cols ) )
        deconv = Deconvolution2D( nb_chns,
                                    self.nb_conv,
                                    self.nb_conv,
                                    self.batch_input_shape,
                                    border_mode='same' )

        reshape_output = Reshape( (nb_rows, nb_cols) )

        # Define decoder
        fc_output_1 = fc_layer_1( self.code )
        fc_output_2 = fc_layer_2( fc_output_1 )
        reshaped_fc_output = reshape_fc( fc_output_2 )
        self.decoder_output = deconv( reshaped_fc_output )
        self.x_rec = reshape_output( self.decoder_output )

        # Define generator
        input_shape = ( self.encode_dim, )
        gen_input = Input( shape=input_shape )

        gen_fc_output_1 = fc_layer_1( gen_input )
        gen_fc_output_2 = fc_layer_2( gen_fc_output_1 )
        gen_reshaped_fc_output = reshape_fc( gen_fc_output_2 )
        gen_decoder_output = deconv( gen_reshaped_fc_output )
        gen_output = reshape_output( gen_decoder_output )

        self.generator = Model(input=[gen_input], output=[gen_output])

    def build_output(self):
        #self.x_rec = K.squeeze( self.decoder_output ,axis=1)
        pass

    def build_loss(self):
        self.cost = objectives.mean_squared_error

    def build_optimizer(self):
        self.optimizer = optimizers.RMSprop(1e-3)

    def build_models(self):
        # For keras models
        self.model = Model(input=[self.x], output=[self.x_rec])
        self.model.compile(loss=self.cost,
                            optimizer=self.optimizer)

        self.encoder = Model(input=[self.x], output=[self.code])

    #########################
    #      Training Ops     #
    #########################
    def train(self, nb_epochs, eval_epoch, save_epoch, result_root):
        exp_dir = os.path.join(result_root,self.data_reader.name, self.name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        training_log_file = os.path.join(exp_dir,'train.log')
        logging.basicConfig(filename=training_log_file,level=logging.INFO)




        logging.info("Begin training")
        for epoch in range(1, nb_epochs+1, 1):
            print("Epoch {}".format(epoch))
            data_generator = self.data_reader.next_batch_generator()
            nb_samples = self.data_reader.nb_samples
            epoch_history = self.model.fit_generator( generator = data_generator ,
                                      samples_per_epoch = nb_samples,
                                      nb_epoch = 1 )

            epoch_loss = epoch_history.history['loss'][0]
            logging.info("Epoch {}, loss{}".format(epoch,epoch_loss))


            if epoch % eval_epoch == 0:
                # Evaluation precision & recall
                labels = self.data_reader.h5_handle['labels'][:]

                code = self.encode()

                prec_list, reca_list = eval_code_pr( labels, code )

                print("Precision:{}\nRecall:{}".format(prec_list,reca_list))

                # save
                if epoch % save_epoch == 0:
                    save_dir = os.path.join(exp_dir,'model_epoch{}'.format(epoch))

                    print("Saving models to {}".format(save_dir))
                    self.save_models_to_dir(save_dir)

                    eval_save_path = os.path.join(exp_dir,'eval_epoch{}.h5'.format(epoch))
                    print("Saving evaluations to {}".format(eval_save_path))
                    evaluation = {
                            'recX': self.reconstruct(),
                            'code': code, # Already encoded
                            'genX': self.generate()
                            }

                    self.save_eval(eval_save_path, evaluation)

    def reconstruct(self):
        logging.debug("Reconstruct")
        recX_data_gen = self.data_reader.next_batch_generator(shuffle=False)
        nb_samples = self.data_reader.nb_samples

        recX = self.model.predict_generator( generator = recX_data_gen,
                                             val_samples = nb_samples )
        return recX

    def encode(self):
        logging.debug("Encode")
        code_data_gen = self.data_reader.next_batch_generator(shuffle=False)
        nb_samples = self.data_reader.nb_samples

        code = self.encoder.predict_generator( generator = code_data_gen,
                                                val_samples = nb_samples )
        return code

    def generate(self):
        if self.encode_dim != 2:
            logging.debug("Encode dimension is not 2, generation skipped")
            return None

        logging.debug("Generate")
        genX_code_gen = self.two_dim_code_generator()
        nb_samples = self.data_reader.nb_samples

        genX = self.generator.predict( genX_code_gen )

        return genX

    def two_dim_code_generator(self, side = 0.5, num = 11 ):
        """
            Work around for generator queue
            TODO: debug
        """
        nx = ny = num
        x_coo = np.linspace(-side, side, nx)
        y_coo = np.linspace(-side, side, ny)

        X, Y = np.meshgrid(x_coo, y_coo)
        # ( 0.5, 0.5), (0.4,0.5)... (-0.4,-0.5),(-0.5,0.5)
        coordinates = np.transpose( np.stack([ X.ravel(), Y.ravel() ]) )
        return coordinates

    ######################
    #     Save & Load    #
    ######################
    def save_models_to_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        logging.debug("Saving model, encoder, generator to {}".format(save_dir))
        model_save_path = os.path.join(save_dir,'model.h5')
        encoder_save_path = os.path.join(save_dir,'encoder.h5')
        generator_save_path = os.path.join(save_dir,'generator.h5')

        self.model.save(model_save_path)
        self.encoder.save(encoder_save_path)
        self.generator.save(generator_save_path)

    def save_eval(self, eval_save_path, evaluation):
        logging.debug("Saving evaluation to {}".format(eval_save_path))
        with h5py.File(eval_save_path,'w') as h5f:
            for k, v in evaluation.iteritems():
                if v is not None:
                    h5f.create_dataset(k, data=v)

    def load_models_from_dir(self, load_dir):
        logging.debug("Loading model, encoder, generator from {}".format(load_dir))
        model_load_path = os.path.join(load_dir,'model.h5')
        encoder_load_path = os.path.join(load_dir,'encoder.h5')
        generator_load_path = os.path.join(load_dir,'generator.h5')

        self.model     = load_model(model_load_path)
        self.encoder   = load_model(encoder_load_path)
        self.generator = load_model(generator_load_path)

class CNNVariationalAutoencoder(CNNAutoencoder):
    def __init__(self,
                 data_reader,
                 hidden_dim,
                 nb_filters,
                 nb_conv,
                 encode_dim,
                 depth,
                 dropout_keep_prob):

        super(CNNVariationalAutoencoder,self).__init__(
                                                 data_reader,
                                                 hidden_dim,
                                                 nb_filters,
                                                 nb_conv,
                                                 encode_dim,
                                                 depth,
                                                 dropout_keep_prob)

    ######################
    #     Build Graph    #
    ######################

    def build_code(self):
        self.z_mean = Dense(self.encode_dim)(self.encoder_output)
        self.z_logvar = Dense(self.encode_dim)(self.encoder_output)
        epsilon_std = 1e-2

        def sampling(args):
            z_mean, z_logvar = args
            batch_size, _, _ = self.batch_input_shape
            epsilon = K.random_normal(shape=(batch_size, self.encode_dim),
                                        mean=0., std=epsilon_std)
            return z_mean + K.exp(z_logvar) * epsilon

        self.z = Lambda(sampling, output_shape=(self.encode_dim,))([self.z_mean, self.z_logvar])

        self.code = self.z

    def build_loss(self):

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)

            rec_loss = objectives.mean_squared_error(x, x_decoded_mean)

            kl_loss = - 0.5 * K.mean(1 + self.z_logvar - K.square(self.z_mean) - K.exp(self.z_logvar), axis=-1)

            return rec_loss + kl_loss

        self.cost = vae_loss

    def build_models(self):
        # For keras models
        self.model = Model(input=[self.x], output=[self.x_rec])
        self.model.compile(loss=self.cost,
                            optimizer=self.optimizer)

        self.encoder = Model(input=[self.x], output=[self.z_mean])

##############################
#   Abbreviations for ease   #
##############################
cnnae  = CNNAE  = CNNAutoencoder
cnnvae = CNNVAE = CNNVariationalAutoencoder

from speech2vec import utils
def get(identifier):
    return utils.get_from_module(identifier,globals(),'cnnautoencoder')
