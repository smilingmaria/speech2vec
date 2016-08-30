
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras import objectives

class KerasBase(BaseAutoencoder):
    def __init__(self,
                data_reader,
                batch_input_shape,
                hidden_dim,
                encode_dim,
                depth,
                nb_filters,
                nb_conv):

        self.data_reader = data_reader
        self.batch_input_shape = batch_input_shape
        self.hidden_dim = hidden_dim
        self.encode_dim = encode_dim
        self.depth = depth

        # For CNN
        self.nb_filters = nb_filters
        self.nb_conv = nb_conv

    @property
    def name(self):
        raise "KerasBase"

    #######################
    #     Build Graph     #
    #######################
    def build_graph(self):
       pass

    #########################
    #      Train & Eval     #
    #########################

    def train(self, nb_epochs):
        data_reader = self.data_reader.next_batch(

    def eval(self):
        pass


