import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import h5py
import numpy as np
from tqdm import tqdm

from utils import makedir

def save_keras_train(model, model_name, epoch, history, result_dir):
    assert result_dir.endswith("/")
    epoch_model_name = model_name + '_epoch_{}'.format(epoch)  

    # Save model json
    json_string = model.to_json()
    open(result_dir + epoch_model_name + '.json','w').write(json_string)
    # Save weights
    model.save_weights(result_dir + epoch_model_name + '_weights.h5',overwrite=True)
    # Save history
    history_string = str(history.history)
    open(result_dir + epoch_model_name + '.history','w').write(history_string)

def save_h5(X, recX, tiledX, y, code, h5path)
    h5f = h5py.File(h5path,'r')
    h5f.create_dataset('X',data=X)
    h5f.create_dataset('recX',data=recX)
    h5f.create_dataset('tiledX',data=tiledX)
    h5f.create_dataset('code',data=code)
    h5f.create_dataset('y',data=y)
    h5f.close()

def save_csv(fbank, csv_dir):
    assert dir_name.endswith("/"), "Add / at the end of csv_path!"
    makedir(csv_dir) 
    
    for idx, arr in tqdm(enumerate(fbank)):
        fname = csv_dir + str(idx+1) + '.csv'
        mask = np.all(np.isnan(arr) | np.equal(arr,0),axis=1)
        arr = arr[~mask]
        np.savetxt(fname,arr,delimiter=",")

def plot_code():
    pass

def plot_generation():
    pass

def plot_tsne(X,Y):
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)

    visX = tsne.fit(X)
    
    plt.figure()
    plt.scatter(visX[:,0], visX[:,1], c=Y, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.show()


