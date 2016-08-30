import logging
import os
import sys

import h5py
import numpy as np

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D
from keras.models import Model, load_model
from keras import backend as K
from keras import objectives, optimizers

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

        self.depth = depth
        print("Depth is currently unused in CNN Autoencoder")

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
        flattened_conv_output = Flatten()(conv_output)

        self.encoder_output = Dense(self.hidden_dim, activation='relu')(flattened_conv_output)

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
    def train(self, nb_epochs, result_root, eval_epoch):
        exp_dir = os.path.join(result_root,self.data_reader.name, self.name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        training_log_file = os.path.join(exp_dir,'train.log')
        logging.basicConfig(filename=training_log_file,level=logging.INFO)

        logging.info("Begin training")

        data_generator = self.data_reader.next_batch_generator()
        nb_samples = self.data_reader.nb_samples

        for epoch in range(1, nb_epochs+1, 1):
            print("Epoch {}".format(epoch))
            epoch_history = self.model.fit_generator( generator = data_generator ,
                                      samples_per_epoch = nb_samples,
                                      nb_epoch = 1 )
            epoch_loss = epoch_history.history['loss'][0]
            logging.info("Epoch {}, loss{}".format(epoch,epoch_loss))
            if eval_epoch & epoch % eval_epoch == 0:

                save_dir = os.path.join(exp_dir,'model_epoch_{}'.format(epoch))
                eval_path = os.path.join(exp_dir,'eval_epoch_{}.h5'.format(epoch))
                self.save(save_dir)
                self.eval(eval_path)


    def eval(self, save_path = None):
        logging.info("Evaluating reconstruction, code, and generation")

        recX_data_gen = self.data_reader.next_batch_generator(shuffle=False)
        code_data_gen = self.data_reader.next_batch_generator(shuffle=False)
        genX_code_gen = self.code_generator()

        nb_samples = self.data_reader.nb_samples

        logging.debug("Evaluating reconstruction...")
        recX = self.model.predict_generator( generator = recX_data_gen,
                                             val_samples = nb_samples )

        logging.debug("Evaluating code...")
        code = self.encoder.predict_generator( generator = code_data_gen,
                                               val_samples = nb_samples )

        if self.encode_dim == 2:
            logging.debug("Evaluating generation...")
            genX = self.generator.predict( genX_code_gen )

        if save_path is not None:
            logging.info("Saving evaluation to {}".format(save_path))
            with h5py.File(save_path,'w') as h5_handle:
                h5_handle.create_dataset('recX',data=recX)
                h5_handle.create_dataset('code',data=code)
                # Evaluate by generator
                if self.encode_dim == 2:
                    h5_handle.create_dataset('genX',data=genX)

        if self.encode_dim == 2:
            return [ recX, code, genX ]
        else:
            return [ recX, code ]

    def code_generator(self, side = 0.5, num = 11 ):
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
    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        logging.debug("Saving model, encoder, generator to {}".format(save_dir))
        model_save_path = os.path.join(save_dir,'model.h5')
        encoder_save_path = os.path.join(save_dir,'encoder.h5')
        generator_save_path = os.path.join(save_dir,'generator.h5')

        self.model.save(model_save_path)
        self.encoder.save(encoder_save_path)
        self.generator.save(generator_save_path)

    def load(self, load_dir):
        logging.debug("Loading model, encoder, generator from {}".format(load_dir))
        model_load_path = os.path.join(load_dir,'model.h5')
        encoder_load_path = os.path.join(load_dir,'encoder.h5')
        generator_load_path = os.path.join(load_dir,'generator.h5')

        self.model     = load_model(model_load_path)
        self.encoder   = load_model(encoder_load_path)
        self.generator = load_model(generator_load_path)
