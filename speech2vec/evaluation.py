from __future__ import division

import logging
import os
import sys


from annoy import AnnoyIndex
import h5py
import numpy as np


def eval_code_pr(labels, code, nbor=10):
    logging.info("Evaluating cluster performance with precision & recall")

    # Build Annoy Tree
    code_dim = code.shape[-1]
    annoy_obj = AnnoyIndex(code_dim)
    for idx, c in enumerate(code):
        annoy_obj.add_item(idx,c)
    annoy_obj.build(10)

    # Define queries
    label_max = np.max(labels)

    prec_list = []
    reca_list = []

    for label in range(label_max):
        query_indices = np.argwhere(labels==label)
        queries = np.squeeze(query_indices).tolist()

        label_retrieved_ids = set()
        for q in queries:
            q_retrieved_ids = annoy_obj.get_nns_by_item(q,
                                                    nbor)
            # Become set and remove self
            q_retrieved_ids = set(q_retrieved_ids)
            q_retrieved_ids.discard(q)

            # Add to current label retrieved total ids
            label_retrieved_ids |= q_retrieved_ids

        # Calculate precision & recall for label

        label_answer_ids = set(queries)

        correct_length = len( label_retrieved_ids & label_answer_ids )

        retrieve_length = len( label_retrieved_ids )
        answer_length   = len(label_answer_ids)


        prec = correct_length / retrieve_length
        reca = correct_length / answer_length

        prec_list.append(prec)
        reca_list.append(reca)

    query_labels = list(range(label_max))
    logging.info("Labels: {}".format(query_labels))
    logging.info("Precision: {}".format(prec_list))
    logging.info("Recall: {}".format(reca_list))
    return prec_list, reca_list

"""
# Function that includes all other functions for saving
def save_reconstruction( sess, model, minloss_modelname, save_dir, dataset ):
    makedir(save_dir)
    batch_size = model.batch_input_shape[0]

    X_rec = model.reconstruct(sess,dataset.next_batch(batch_size=batch_size, shuffle = False))
    code  = model.encode(sess,dataset.next_batch(batch_size=batch_size, shuffle = False))

    X_rec = dataset.fit_X_shape(X_rec)
    code  = dataset.fit_X_shape(code)

    h5_path = save_dir + minloss_modelname + '.h5'

    print "Saving feature and yphase to %s" % h5_path
    save_h5( h5_path, X_rec, code )
    feature_path = save_dir + dataset.data_type + '/'
    print "Saving feature to %s" % feature_path
    save_to_csv( feat, feature_path)

    yphase_path = save_dir + 'yphase/'
    print "Saving yphase to %s" % yphase_path
    save_to_csv( phase, yphase_path )

def save_h5(h5_path, recX, code):
    with h5py.File(h5_path, 'w') as h5f:
        h5f.create_dataset("recX",data=recX)
        #h5f.create_dataset("yphase",data=yphase)
        h5f.create_dataset("code",data=code)

def save_to_csv( arr,dir_name):
    assert dir_name.endswith("/")
    makedir(dir_name)

    for idx,arr in tqdm(enumerate(arr)):
        fname = dir_name + str(idx+1) + ".csv"
        # Convert Nan to zeros
        arr = np.nan_to_num(arr)
        # Remove rows that are all nans or all zeros
        #mask = np.all(np.isnan(arr) | np.equal(arr, 0), axis=1)
        #arr = arr[~mask]
        np.savetxt(fname,arr,delimiter=",")
"""
