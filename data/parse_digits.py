from collections import defaultdict
import cPickle as pickle
import csv
import h5py
import os
import shutil

zero  = 199479
one   = 125608 
two   = 182873
three = 176922
four  = 65278
five  = 62860 
six   = 161986
seven = 158517
eight = 53844
nine  = 122081
ten   = 174938

digit2word = {
    0:zero,
    1:one,
    2:two,
    3:three,
    4:four,
    5:five,
    6:six,
    7:seven,
    8:eight,
    9:nine,
    10:ten
    }


word2digit = { v: k for k, v in digit2word.iteritems() }

map_file = './dev-clean/fbank.map'
flac_dir = './dev-clean/flacs/'

new_flac_dir = './digits/flacs/'

digit_count = defaultdict(int)

with open(map_file, 'r') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        order, word_id, flac_name = row
        word_id = int(word_id)
        if word_id in word2digit:
            digit = word2digit[ word_id ]
            flac_file = flac_dir + flac_name
            new_flac_file = new_flac_dir + flac_name

            shutil.copyfile(flac_file, new_flac_file)
            digit_count[ digit ] += 1
print map_file
print digit_count
