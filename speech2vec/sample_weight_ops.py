import numpy as np


def loss_fnt_masks(X, mask_value=0.):
    """
        X is a 3 dim tuple ( sample, timesteps, feature )
    """
    return np.any(np.not_equal(X, mask_value), axis=-1).astype('float32')

if __name__ == "__main__":
    a = np.ones((5,4,2))
    b = np.zeros((5,3,2))

    c = np.hstack((a,b))

    
    d = np.ones((5,4))
    e = np.zeros((5,3))
    f = np.hstack((d,e))

    assert ( f == loss_fnt_masks(c) ) .all()
