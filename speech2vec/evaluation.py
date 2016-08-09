import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import h5py
import numpy as np
from tqdm import tqdm

def plot_tsne(X,Y):
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)

    visX = tsne.fit(X)
    
    plt.figure()
    plt.scatter(visX[:,0], visX[:,1], c=Y, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.show()


