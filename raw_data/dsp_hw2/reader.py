from collections import defaultdict
import csv
from glob import glob
import os
import pickle
import pprint
import re
import sys

ch2num = {
        'i': 1,
        'er': 2,
        'san': 3,
        'sy': 4,
        'u': 5,
        'liou': 6,
        'qi': 7,
        'ba': 8,
        'jiou': 9,
        'liN': 0,
        'yi': 1,
        'si': 4,
        'wu': 5,
        'liu': 6,
        'qi': 7,
        'jiu': 9,
        'ling': 0
        }

def run_label_parsing(label_dir, label_pickle):
    glob_path = os.path.join(label_dir,'*.mlf')

    label_dict = defaultdict(list)
    lab_pat = r'N(\d+).lab'
    if not os.path.exists(label_pickle):
        print("Creating label pickle at {} ".format(label_pickle))
        for fname in glob( glob_path ):
            with open(fname,'r') as f:
                cur_wavname = ''
                for line in f.readlines():
                    row = line.strip().lstrip('#')
                    # MLF header
                    if "MLF" in row:
                        continue

                    # wav name
                    if '.lab' in row:
                        wav_lab = re.search(lab_pat, row).group(0)
                        wav_name = wav_lab.rstrip('.lab')
                        cur_wavname = wav_name

                    if row in ch2num.keys():
                        label_dict[ cur_wavname ].append( ch2num[row] )

        with open(label_pickle,'wb') as f:
            pickle.dump( label_dict, f)
    else:
        print("{} already exists".format(label_pickle))

def create_feature_map( label_pickle, wav_dir, feature_map_path ):
    if not os.path.exists(feature_map_path):
        print("Creating feature map at {}".format(feature_map_path))
        with open(label_pickle,'rb') as f:
            label_dict = pickle.load(f)

        glob_path = os.path.join(wav_dir,'*.wav')
        wav_pat = 'N(\d+).wav'

        rows_to_write = []

        for idx,wav_path in enumerate(sorted( glob( glob_path ) )):
            wav_filename = re.search(wav_pat, wav_path).group(0)
            wav_name = wav_filename.rstrip('.wav')

            digit = label_dict[ wav_name ]

            digit = set(digit)
            assert len(digit) == 1
            digit = digit.pop()

            gender = int( wav_name[2] )

            row = [ idx, wav_filename, digit, gender ]

            rows_to_write.append(row)

        with open(feature_map_path,'wb') as fout:
            csvwriter = csv.writer(fout, delimiter=',')
            csvwriter.writerows( rows_to_write )
    else:
        print("Feature map already exists at {}".format(feature_map_path))

def calculate_stats( feature_map_path ):
    digit_gender_count = defaultdict(int)

    with open(feature_map_path,'r') as f:
        reader = csv.reader(f,delimiter=',')
        for row in reader:
            digit = int(row[2])
            gender = int(row[3])

            if gender == 1:
                digit_gender_count[ str(digit) + "_male" ] += 1
            elif gender == 2:
                digit_gender_count[ str(digit) + "_female" ] += 1
            else:
                raise ValueError("Unexpected gender number {}".format(gender))
    keys = sorted(digit_gender_count.keys())
    for k in keys:
        print("{} : {}".format(k, digit_gender_count[k]))

if __name__ == "__main__":
    label_dir = 'raw/labels/'
    label_pickle = 'labels.pkl'

    wav_dir = './wavs/'
    feature_map_path = './feature.map'

    run_label_parsing(label_dir, label_pickle)

    create_feature_map( label_pickle, wav_dir, feature_map_path )

    calculate_stats(feature_map_path)
