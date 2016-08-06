import numpy as np
from annoy import AnnoyIndex

def build_annoy_tree( X ):
    # X is a 3D array
    sample, timestep, feature = X.shape

    # Reshape to all sizes
    X = X.reshape( sample * timestep, feature)
    # Remove duplicates
    X = np.vstack({tuple(row) for row in X})

    ann = AnnoyIndex(feature)

    for idx, feat in enumerate(X):
        ann.add_item(idx, feat)

    ann.build(10)

    return ann

def tile_reconstruction(recX, ann):
    def get_closest_vec_from_annoy(x, ann):
        idx = ann.get_nns_by_vector(x, n=1)
        x_hat = ann.get_item_vector(idx)
        return x_hat

    tiled_X = []
    for sample in recX:
        audio = []
        for feature in sample:
            tiled_feature = get_closest_vec_from_annoy(feature,ann)
            audio.append(feature)
        audio = np.array(audio)
        tiled_X.append(audio)
    tiled_X = np.array(tiled_X)
    return tiled_X
