### Pipeline Work flow

---
* Import tensorflow and get project root path
```python
import tensorflow as tf
import config # See config.py 
```
* Select dataset to import, also load data set paramters
```python
from speech2vec.datasets import dsp_hw2 as load_dataset

# Load data
dataset = load_dataset()
result_dir = '../result/' + load_dataset.__name__ + '/'

# Data samples
X = dataset.X
y = dataset.y

# Get Shape
sample, timestep, feature = X.shape
```

* Select model, specify model architecture and param
```python
from speech2vec.models import Seq2seqAutoencoder

cells = ['GRUCell'] * 2

nb_epochs = 1
batch_size = 64
hidden_dim = 128
depth = (1,1)
keep_prob = 0.8
peek = False
bidirectional = False

batch_input_shape = ( batch_size, timestep, feature )

# Build model
model = Seq2seqAutoencoder( batch_input_shape,\
                            cells,\
                            hidden_dim,\
                            depth,\
                            keep_prob,\
                            peek = peek,\
                            bidirectional=bidirectional)
# Build graph is important!
model.build_graph()

# Name is used for saving
model_name = model.name
```
* Start session and start training
```python

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    # Intialize variables
    tf.initialize_all_variables().run()
    # Run every epoch
    min_loss = sys.float_info.max
    for epoch in range(1, nb_epochs+1, 1):
        # train_one_epoch is defined in models
        epoch_loss = model.train_one_epoch( sess, dataset.next_batch(batch_size=batch_size) )
        
        # Save the best model
        if epoch_loss < min_loss: 
            min_loss = epoch_loss
            save_path = result_dir + model_name + '.ckpt'
            model.save(sess, saver, save_path)
        
        print "Epoch {}, loss {}, min_loss {}".format( epoch, epoch_loss, min_loss)
        print "Test loss", model.test(sess, dataset.next_batch(batch_size=batch_size))
```
* Reconstruct & Encode ( or Generate ) 
```python
    
    load_path = save_path
    model.load(sess, saver, load_path)

    X_rec = model.reconstruct(sess,dataset.next_batch(batch_size=batch_size, shuffle = False))
    code  = model.encode(sess,dataset.next_batch(batch_size=batch_size, shuffle = False)) 
  
    feat, phase = dataset.split_X(X_rec)

    save_h5( h5_path, feat, phase, code)
```
