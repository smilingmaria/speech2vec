import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import h5py
import numpy as np
from tqdm import tqdm

def save_h5(h5_path, feature, yphase, code):
    with h5py.File(h5_path, 'w') as h5f:
        h5f.create_dataset("feature",data=feature)
        h5f.create_dataset("yphase",data=yphase)
        h5f.create_dataset("code",data=code)

def load_h5(h5_path):
    with h5py.File(h5_path,'r') as h5f:
        X_rec = h5f['X_rec'][:]
        code  = h5f['code'][:]
    return X_rec, code

def plot_tsne(X,Y):
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)

    visX = tsne.fit(X)
    
    plt.figure()
    plt.scatter(visX[:,0], visX[:,1], c=Y, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.show()


