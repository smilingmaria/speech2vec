from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras import objectives, optimizers

class CNNAutoencoder(object):
    def __init__(self,
                 data_reader,
                 batch_input_shape,
                 hidden_dim,
                 nb_filters,
                 nb_conv,
                 encode_dim,
                 depth,
                 dropout_keep_prob):

        self.data_reader = data_reader

        self.batch_input_shape = batch_input_shape
        self.hidden_dim = hidden_dim
        self.nb_filters = nb_filters
        self.nb_conv = nb_conv
        self.encode_dim = encode_dim

        self.depth = depth
        print("Depth is currently unused in CNN Autoencoder")

        self.dropout_keep_prob = dropout_keep_prob

    @property
    def name(self):
        namestring = self.__class__.__name__
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

        # For keras models
        self.model = Model(input=[self.x], output=[self.x_rec])
        self.encoder = Model(input=[self.x], output=[self.code])
        #self.generator = Model(input=[self.code],output=[self.x_rec])

    def build_inputs(self):
        self.x = Input(batch_shape = self.batch_input_shape)

    def build_encoder(self):
        flattened_conv_output = Flatten()(Convolution2D( self.nb_filters,
                                            self.nb_conv,
                                            self.nb_conv,
                                            border_mode='same',
                                            activation='relu')(self.x))

        self.encoder_output = Dense(self.hidden_dim, activation='relu')(flattened_conv_output)

    def build_code(self):
        self.code = Dense(self.encode_dim, activation='relu')(self.encoder_output)

    def build_decoder(self):
        batch_size, nb_chns, nb_rows, nb_cols = self.batch_input_shape

        # Define layers
        fc_layer_1 = Dense(self.hidden_dim, activation='relu')
        fc_layer_2 = Dense(self.nb_filters * nb_rows * nb_cols, activation='relu')

        reshape_fc = Reshape( ( self.nb_filters, nb_rows, nb_cols ) )
        deconv = Deconvolution2D( nb_chns,
                                    self.nb_conv,
                                    self.nb_conv,
                                    self.batch_input_shape,
                                    border_mode='same' )

        # Define decoder
        fc_output_1 = fc_layer_1( self.code )
        fc_output_2 = fc_layer_2( fc_output_1 )
        reshaped_fc_output = reshape_fc( fc_output_2 )
        self.decoder_output = deconv( reshaped_fc_output )

        # Define generator
        gen_input = Input( batch_shape = ( batch_size, self.encode_dim ) )

        gen_fc_output_1 = fc_layer_1( gen_input )
        gen_fc_output_2 = fc_layer_2( gen_fc_output_1 )
        gen_reshaped_fc_output = reshape_fc( gen_fc_output_2 )
        gen_decoder_output = deconv( gen_reshaped_fc_output )
        gen_output = gen_decoder_output

        self.generator = Model(input=[ gen_input], output=[gen_output])

    def build_output(self):
        self.x_rec = self.decoder_output

    def build_loss(self):
        self.cost = objectives.mean_squared_error

    def build_optimizer(self):
        self.optimizer = optimizers.RMSprop(1e-3)

    #########################
    #      Training Ops     #
    #########################
    def train(self, nb_epochs):

        data_generator = self.data_reader.next_batch_generator()
        nb_samples = self.data_reader.nb_samples

        self.model.fit_generator(  generator = data_generator ,
                                   samples_per_epoch = nb_samples,
                                   nb_epoch = nb_epochs )

    def eval(self):
        recX_data_gen = self.data_reader.next_batch_generator(shuffle=False)
        code_data_gen = self.data_reader.next_batch_generator(shuffle=False)
        genX_code_gen = self.code_generator()

        nb_samples = self.data_reader.nb_samples

        recX = self.model.predict_generator( generator = recX_data_gen,
                                             val_samples = nb_samples )

        code = self.encoder.predict_generator( generator = code_data_gen,
                                               val_samples = nb_samples )

        genX = self.generator.predict_generator( generator = genX_code_gen,
                                               val_samples = 121 )
        return [ recX, code, genX ]

    def code_generator(self, side = 0.5, num = 11 ):
        nx = ny = num
        x_coo = np.linspace(-side, side, nx)
        y_coo = np.linspace(-side, side, ny)

        X, Y = np.meshgrid(x_coo, y_coo)
        # ( 0.5, 0.5), (0.4,0.5)... (-0.4,-0.5),(-0.5,0.5)
        coordinates = np.transpose( np.stack([ X.ravel(), Y.raveL() ]) )

        for coo in coordinates:
            yield coo

if __name__ == "__main__":
    data_reader = None
    cnn_ae = CNNAutoencoder( data_reader,
                             batch_input_shape = ( 32, 1, 60, 75 ),
                             hidden_dim = 128,
                             nb_filters = 32,
                             nb_conv = 3,
                             encode_dim = 10,
                             depth = (1,1),
                             dropout_keep_prob = 0.8)

    cnn_ae.build_graph()

    print cnn_ae.summary()
    import pdb; pdb.set_trace()
