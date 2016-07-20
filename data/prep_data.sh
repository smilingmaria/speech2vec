#!/bin/bash
raw_data=
if [ -v raw_data ]; then 
	echo "Please specify where you want LibriSpeech data to be downloaded to";
	exit 
fi

echo "Make sure you have around 80G space on disk!"
echo "This might take a while..."

# Download LibriSpeech data, script taken from kaldi
raw_data_url=www.openslr.org/resources/12

for part in dev-clean test-clean; do
	./download_and_untar.sh $raw_data $raw_data_url $part
done

# Cut data
echo "Cutting flacs according to alignments..."
for part in dev-clean test-clean; do
	mkdir -p ./$part
	./cut_frames.sh $raw_data/LibriSpeech/$part ./$part
	mkdir ./$part/flacs
	mkdir ./$part/fbank
	mkdir ./$part/fbank_delta
	mkdir ./$part/yphase
done;	

# Run matlab code
cd ../fbank_matlab
echo "Extracting fbank with matlab & writing to csv files..."
for part in dev-clean test-clean; do
	path_to_dir=../data/$part/
	matlab -nodisplay -nojvm -r "extract_fbank_from_flac($path_to_dir)"
done;
cd ../data

# Store in h5py Format
echo "Store to h5py..."
for part in dev-clean test-clean; do
	path_to_dir=./$part/
	python ../util/dataset_ops.py -a create -p $path_to_dir
done;

echo "Now you can start running training scripts"

