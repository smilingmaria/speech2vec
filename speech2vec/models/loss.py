import keras.backend as K

def reverse_mse(reverse=False):
    def mean_squared_error( y_true, y_pred ):
        if reverse:
            return K.mean( K.square( y_pred[:,:,::-1] - y_true ), axis=-1)
        else:
            return K.mean( K.square( y_pred - y_true), axis=-1)
    return mean_squared_error
