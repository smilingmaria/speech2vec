import numpy as np

class Scaler(object):
    def __init__(self):
        raise NotImplementedError

    def fit_transform(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError
    
    def transform(self):
        raise NotImplementedError

    def inverse_transform(self):
        raise NotImplementedError

class InstanceWise(Scaler):
    def __init__(self):
        pass

    def fit_transform(self, arr):
        self.fit(arr)
        norm_arr = self.transform(arr)
        return norm_arr

    def fit(self, arr):
        self._norm = np.linalg.norm(arr, axis=-1)[...,None]

    def transform(self, arr):
        norm_arr = arr / self._norm
        norm_arr = np.nan_to_num( norm_arr ) 
        return norm_arr

    def inverse_transform(self, norm_arr):
        return norm_arr * self._norm

class FeatureWise(Scaler):
    def __init__(self):
        pass

    def fit_transform(self, arr):
        self.fit(arr)
        norm_arr = self.transform(arr)
        return norm_arr

    def fit(self, arr):
        self._mean = arr.mean(axis=0)
        self._std  = arr.std(axis=0)
    
    def transform(self, arr):
        norm_arr = ( arr - self._mean ) / self._std
        norm_arr = np.nan_to_num( norm_arr )
        return arr

    def inverse_transform(self, norm_arr):
        arr = ( norm_arr * self._std ) + self._mean
        return arr

