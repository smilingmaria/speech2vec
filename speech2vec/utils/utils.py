import os
import numpy as np

def makedir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
                raise

if __name__ == "__main__":
    _, X, _ = load_digits()

    ann = build_annoy_tree(X)
