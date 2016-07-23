# -*- coding: utf-8 -*-
"""
	TODO: onetomanySimpleRNN, onetomanyGRU
"""

from __future__ import absolute_import
import copy

import numpy as np

from keras import backend as K
from keras import activations, initializations
from keras.engine import InputSpec
from keras.layers import SimpleRNN, GRU, LSTM

class onetomanySimpleRNN(SimpleRNN):
    def __init__(self, output_length, output_dim, **kwargs):
        super(onetomanySimpleRNN, self).__init__(output_dim,return_sequences=True,**kwargs)
        self.output_length = output_length
        # For 2D input
        self.input_spec = [ InputSpec(ndim=2) ]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[1]
        self.input_dim = input_dim

        if self.stateful:
            self.reset_states()
        else:
            self.states = [None, None]

        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        self.U = self.inner_init((self.output_dim, self.output_dim),
                                 name='{}_U'.format(self.name))
        self.b = K.zeros((self.output_dim,), name='{}_b'.format(self.name))

        # Weights for output to input
        self.W_y = self.init((self.output_dim, self.input_dim),
                             name='{}_W_y'.format(self.name))

        self.b_y = K.zeros((self.input_dim,), name='{}_b_y'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
            self.regularizers.append(copy.deepcopy(self.W_regularizer).set_param(self.W_y))
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        
	if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
            self.regularizers.append(copy.deepcopy(self.b_regularizer).set_param(self.b_y))

        self.trainable_weights = [self.W, self.U, self.b,
                                  self.W_y, self.b_y]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights


    def call(self, x, mask=None):
        X = K.repeat(x, self.output_length)

        input_shape = self.input_spec[0].shape

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)

        constants = self.get_constants(X)
        preprocessed_input = self.preprocess_input(X)

        initial_states[1] = x

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=self.output_length)
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.input_dim))]

    def preprocess_input(self, x):
        return x

    def step(self, x, states):
        prev_output = states[0]
        y_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]

        h = K.dot(y_tm1 * B_W[0], self.W) + self.b

        output = self.activation(h + K.dot(prev_output * B_U, self.U))
        y = self.activation(K.dot(output * B_W[1], self.W_y) + self.b_y)

        return output, [output, y]

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        # x here is of shape (samples, input_dim)
        initial_state = K.zeros_like(x)
        initial_state = K.sum(initial_state, axis=1) # (samples,)
        initial_state = K.expand_dims(initial_state) # (sample,1)
        initial_state = K.tile(initial_state, [1, self.output_dim])
        initial_state_y = K.tile(initial_state, [1, self.input_dim])

        initial_states = [ initial_state for _ in range(len(self.states)-1) ] + \
                            [ initial_state_y ]

        return initial_states

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
            constants.append(B_U)
        else:
            constants.append(K.cast_to_floatx(1.))

        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [ K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(2)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(2)])
        return constants

    def get_config(self):
        config = {'output_length': self.output_length}
        base_config = super(onetomanySimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class onetomanyGRU(GRU):
    def __init__(self, output_length, output_dim, **kwargs):
        super(onetomanyGRU, self).__init__(output_dim,return_sequences=True,**kwargs)
        self.output_length = output_length
        # For 2D input
        self.input_spec = [ InputSpec(ndim=2) ]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[1]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None, None]

        if self.consume_less == 'gpu':

            self.W = self.init((self.input_dim, 3 * self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, 3 * self.output_dim),
                                     name='{}_U'.format(self.name))

            self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                           np.zeros(self.output_dim),
                                           np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))

            # Weights for output to input
            self.W_y = self.init((self.output_dim, self.input_dim),
                                 name='{}_W_y'.format(self.name))

            self.b_y = K.zeros((self.input_dim,), name='{}_b_y'.format(self.name))

            self.trainable_weights = [self.W, self.U, self.b,
                                      self.W_y, self.b_y]
        else:

            self.W_z = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_z'.format(self.name))
            self.U_z = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_z'.format(self.name))
            self.b_z = K.zeros((self.output_dim,), name='{}_b_z'.format(self.name))

            self.W_r = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_r'.format(self.name))
            self.U_r = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_r'.format(self.name))
            self.b_r = K.zeros((self.output_dim,), name='{}_b_r'.format(self.name))

            self.W_h = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_h'.format(self.name))
            self.U_h = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_h'.format(self.name))
            self.b_h = K.zeros((self.output_dim,), name='{}_b_h'.format(self.name))

            # Weights for output to input
            self.W_y = self.init((self.output_dim, self.input_dim),
                                 name='{}_W_y'.format(self.name))
            
	    self.b_y = K.zeros((self.input_dim,), name='{}_b_y'.format(self.name))

            self.trainable_weights = [self.W_z, self.U_z, self.b_z,
                                      self.W_r, self.U_r, self.b_r,
                                      self.W_h, self.U_h, self.b_h,
                                      self.W_y, self.b_y]

            self.W = K.concatenate([self.W_z, self.W_r, self.W_h])
            self.U = K.concatenate([self.U_z, self.U_r, self.U_h])
            self.b = K.concatenate([self.b_z, self.b_r, self.b_h])

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
            self.regularizers.append(copy.deepcopy(self.W_regularizer).set_param(self.W_y))
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
            self.regularizers.append(copy.deepcopy(self.b_regularizer).set_param(self.b_y))

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        X = K.repeat(x, self.output_length)

        input_shape = self.input_spec[0].shape
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)

        constants = self.get_constants(X)
        preprocessed_input = self.preprocess_input(X)

        initial_states[1] = x

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=self.output_length)
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
                return last_output

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                            K.zeros((input_shape[0], self.input_dim))]

    def preprocess_input(self, x):
        return x

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        y_tm1 = states[1]
        B_U = states[2]  # dropout matrices for recurrent units
        B_W = states[3]

        if self.consume_less == 'gpu':

            matrix_x     = K.dot(y_tm1 * B_W[0], self.W) + self.b
            matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.output_dim])

            x_z = matrix_x[:, :self.output_dim]
            x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
            inner_z = matrix_inner[:, :self.output_dim]
            inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.output_dim:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.output_dim:])
            hh = self.activation(x_h + inner_h)
        else:
            if self.consume_less == 'cpu' or self.consume_less == 'mem':
                x_z = K.dot(y_tm1 * B_W[0], self.W_z) + self.b_z
                x_r = K.dot(y_tm1 * B_W[1], self.W_r) + self.b_r
                x_h = K.dot(y_tm1 * B_W[2], self.W_h) + self.b_h
            else:
                raise Exception('Unknown `consume_less` mode.')
            z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
            r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

            hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))
        h = z * h_tm1 + (1 - z) * hh

        y = self.activation(K.dot(h * B_W[3], self.W_y) + self.b_y)

        return h, [h,y]


    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        # x here is of shape (samples, input_dim)
        initial_state   = K.zeros_like(x)
        initial_state   = K.sum(initial_state, axis=1) # (samples,)
        initial_state   = K.expand_dims(initial_state) # (sample,1)
        initial_state   = K.tile(initial_state, [1, self.output_dim])
        initial_state_y = K.tile(initial_state, [1, self.input_dim])

        initial_states = [ initial_state for _ in range(len(self.states)-1) ] + \
                            [ initial_state_y ]

        return initial_states

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(3)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def get_config(self):
        config = {'output_length': self.output_length}
        base_config = super(onetomanyGRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class onetomanyLSTM(LSTM):
    def __init__(self, output_length, output_dim, **kwargs):
        super(onetomanyLSTM, self).__init__(output_dim,return_sequences=True,**kwargs)
        self.output_length = output_length
        # For 2D input
        self.input_spec = [ InputSpec(ndim=2) ]

    def build(self, input_shape):
        # input_shape should be a 2D tensor
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[1]

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None, None, None]

        if self.consume_less == 'gpu':
            self.W = self.init((self.input_dim, 4 * self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, 4 * self.output_dim),
                                     name='{}_U'.format(self.name))

            self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                           K.get_value(self.forget_bias_init((self.output_dim,))),
                                           np.zeros(self.output_dim),
                                           np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))

            # Weights for output to input
            self.W_y = self.init((self.output_dim, self.input_dim),
                                 name='{}_W_y'.format(self.name))

            self.b_y = K.zeros((self.input_dim,), name='{}_b_y'.format(self.name))

            self.trainable_weights = [self.W, self.U, self.b,
                                    self.W_y, self.b_y]
        else:
            self.W_i = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_i'.format(self.name))
            self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_i'.format(self.name))
            self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

            self.W_f = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_f'.format(self.name))
            self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_f'.format(self.name))
            self.b_f = self.forget_bias_init((self.output_dim,),
                                             name='{}_b_f'.format(self.name))

            self.W_c = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_c'.format(self.name))
            self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_c'.format(self.name))
            self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

            self.W_o = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_o'.format(self.name))
            self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_o'.format(self.name))
            self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

            # Weights for output to input
            self.W_y = self.init((self.output_dim, self.input_dim),
                                 name='{}_W_y'.format(self.name))

            self.b_y = K.zeros((self.input_dim,), name='{}_b_y'.format(self.name))


            self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                      self.W_c, self.U_c, self.b_c,
                                      self.W_f, self.U_f, self.b_f,
                                      self.W_o, self.U_o, self.b_o,
                                      self.W_y, self.b_y]

            self.W = K.concatenate([self.W_i, self.W_f, self.W_c, self.W_o])
            self.U = K.concatenate([self.U_i, self.U_f, self.U_c, self.U_o])
            self.b = K.concatenate([self.b_i, self.b_f, self.b_c, self.b_o])

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
            self.regularizers.append(copy.deepcopy(self.W_regularizer).set_param(self.W_y))

        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
            self.regularizers.append(copy.deepcopy(self.b_regularizer).set_param(self.b_y))

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def preprocess_input(self, x):
        return x

    def call(self, x, mask=None):
        X = K.repeat(x, self.output_length)

        input_shape = self.input_spec[0].shape

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)

        constants = self.get_constants(X)
        preprocessed_input = self.preprocess_input(X)

        initial_states[2] = x

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=self.output_length)
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], self.input_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.input_dim))]

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        y_tm1 = states[2]
        B_U = states[3]
        B_W = states[4]

        if self.consume_less == 'gpu':
            z = K.dot(y_tm1 * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U) + self.b

            z0 = z[:, :self.output_dim]
            z1 = z[:, self.output_dim: 2 * self.output_dim]
            z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
            z3 = z[:, 3 * self.output_dim:]

            i = self.inner_activation(z0)
            f = self.inner_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.inner_activation(z3)
        else:
            if self.consume_less == 'mem' or self.consume_less == "cpu":
                x_i = K.dot(y_tm1 * B_W[0], self.W_i) + self.b_i
                x_f = K.dot(y_tm1 * B_W[1], self.W_f) + self.b_f
                x_c = K.dot(y_tm1 * B_W[2], self.W_c) + self.b_c
                x_o = K.dot(y_tm1 * B_W[3], self.W_o) + self.b_o
            else:
                raise Exception('Unknown `consume_less` mode.')

            i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
            f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
            o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))


        h = o * self.activation(c)

        # Feed output back to input
        y = self.activation(K.dot(h * B_W[4], self.W_y) + self.b_y)

        return h, [h, c, y]

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        # x here is of shape (samples, input_dim)
        initial_state = K.zeros_like(x)
        initial_state = K.sum(initial_state, axis=1) # (samples,)
        initial_state = K.expand_dims(initial_state) # (sample,1)
        initial_state = K.tile(initial_state, [1, self.output_dim])
        initial_state_y = K.tile(initial_state, [1, self.input_dim])

        initial_states = [ initial_state for _ in range(len(self.states)-1) ] + \
                            [ initial_state_y ]

        return initial_states

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(5)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(5)])
        return constants

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_length, self.output_dim)

    def get_config(self):
        config = {'output_length': self.output_length}
        base_config = super(onetomanyLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
