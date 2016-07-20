import os

def makedir(path):
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
                raise
