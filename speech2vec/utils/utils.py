
from __future__ import absolute_import

try:
    import cPickle as pickle
except ImportError:
    import pickle

import six

"""
    Useful utility, took from keras
"""
def get_from_module(identifier, module_params, module_name,
                    instantiate=False, kwargs=None):
    if isinstance(identifier, six.string_types):
        res = module_params.get(identifier)
        if not res:
            raise Exception('Invalid ' + str(module_name) + ': ' +
                            str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    elif type(identifier) is dict:
        name = identifier.pop('name')
        res = module_params.get(name)
        if res:
            return res(**identifier)
        else:
            raise Exception('Invalid ' + str(module_name) + ': ' +
                            str(identifier))
    return identifier


def save_to_pickle(save_path, obj):
    with open(save_path,'wb') as f:
        pickle.dump(obj, f)

def load_from_pickle(load_path):
    with open(load_path,'rb') as f:
        return pickle.load(f)
