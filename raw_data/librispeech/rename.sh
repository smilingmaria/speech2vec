#! /bin/bash

for i in {63904..1}; do
	let j=$i-1
	echo "mv $j.csv $i.csv"
	mv $j.csv $i.csv
done;
