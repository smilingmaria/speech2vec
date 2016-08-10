import h5py

def train_one_epoch( sess, model, keep_prob, dataset ):

    batch_size, timestep, feature = model.batch_input_shape

    epoch_loss = 0.
    batch_coutner = 0
    for x, y in dataset.next_batch(batch_size=batch_size):
        feed = { model.x : x, model,keep_prob: keep_prob }
        loss, _ = sess.run([ model.cost, model.optimizer ], feed_dict = feed )

        epoch_loss += loss
        batch_counter += 1
    epoch_loss /= batch_counter

    # Print info
    return epoch_loss

def predict(sess, model, dataset):
    batch_size, timestep, feature = model.batch_input_shape
    
    sample = model.X.shape[0]
    
    toadd = ( batch_size - sample % batch_size ) % batch_size
    
    X = np.vstack(model.X, model.X[:toadd])

    X_rec = []
    for idx in range(0, X.shape[0], batch_size)
        x = X[idx:idx+batch_size]
        feed = { model.x : x, model,keep_prob: keep_prob }
        x_rec = sess.run([ model.x_rec ], feed_dict = feed )
        X_rec.append(x_rec)

    X_rec = np.vstack( X_rec ) 
    X_rec = X_rec[:sample]
    
    return X_rec

def save_prediction(h5f_name, fbank, yphase):
    h5f = h5py.File(h5f_name,'w')
     
