#! /bin/bash
if [ -z "$3" ] ; then
	echo 'Supply 3 arguments'
	echo "[alignment file] [raw_data directory] [save data directory]"
	exit 1
fi

echo "ffmpeg used with log level quiet, meaning non verbose"

word_in_utter_counter=1
prev_flac_id=""
while read line; do
	# Read lines
	read flac_id raw_time_start raw_duration word_id <<< $line
	IFS='-' read first_prefix second_prefix id <<< $flac_id
	#echo $first_dir $second_dir
	# Define outfile name
	infile=$2/$first_prefix/$second_prefix/$flac_id.flac
	time_start=$(awk "BEGIN {printf \"%.2f\",${raw_time_start}/100}")
	duration=$(awk "BEGIN {printf \"%.2f\",${raw_duration}/100}")
	outfile=$3/${flac_id}_${word_id}_${word_in_utter_counter}.flac
	
	# Split using ffmpeg	
	ffmpeg -loglevel error -i $infile -ss $time_start -t $duration $outfile < /dev/null
	
	# For word counter
	if [[ -z "$prev_flac_id" ]]; then
		echo -ne "\rProcessing $flac_id..."
	fi
	if [[ -z "$prev_flac_id" ]] || [[ $flac_id = $prev_flac_id ]]; then
		let word_in_utter_counter=word_in_utter_counter+1
	else
		echo -ne "\nProcessing $flac_id...\r"
		let word_in_utter_counter=1
	fi
	prev_flac_id=$flac_id
done < $1	
echo -ne "\nDone with $1 $2 $3\n"
