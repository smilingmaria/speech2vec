import tensorflow as tf

class BaseAutoencoder(object):
    def __init__(self,
                data_reader,
                batch_input_shape,
                hidden_dim,
                encode_dim,
                depth):

        self.data_reader = data_reader
        self.batch_input_shape = batch_input_shape
        self.hidden_dim = hidden_dim
        self.encode_dim = encode_dim
        self.depth = depth

    @property
    def name(self):
        return "BaseAutoencoder"

    #######################
    #   Build Feed List   #
    #######################
    def build_feed_dict(self):
        pass

    def build_feed_list(self):
        pass

    #######################
    #     Build Graph     #
    #######################
    def build_graph(self):
        self.build_inputs()
        self.build_encoder()
        self.build_code()
        self.build_decoder()
        self.build_loss()
        self.build_optimizer()

    def build_inputs(self):
        raise NotImplementedError

    def build_encoder(self):
        raise NotImplementedError

    def build_code(self):
        raise NotImplementedError

    def build_decoder(self):
        raise NotImplementedError

    def build_loss(self):
        raise NotImplementedError

    def build_optimizer(self):
        raise NotImplementedError

    def build_optimizer(self):
        raise NotImplementedError
