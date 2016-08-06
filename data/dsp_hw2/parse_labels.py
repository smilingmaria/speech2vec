from collections import defaultdict
from glob import glob
import cPickle as pickle

answer_files = glob('./labels/*.mlf')

chi2en = {
	'sil': -1, 
	'ling': 0,
	'yi': 1,
	'er': 2,
	'san': 3,
	'si': 4, 
	'wu': 5,
	'liu': 6, 
	'qi': 7,
	'ba': 8,
	'jiu': 9
}


labels = defaultdict(list)

def is_wavname( line ):
	line = line.strip('\n\"*/')
	return line.endswith('.lab')

def is_period(line):
	return line == '.'

if __name__ == "__main__":
	for filename in answer_files:
		print "Reading", filename
		with open(filename,'r') as fin:
			lines = fin.readlines() # Since data set is small enough
			lines = map(lambda x: x.strip('\n#'), lines[1:])	
			idx = 0
			while idx < len(lines):
				line = lines[idx]
				if is_wavname(line):
					wav_name = line.rstrip('.lab')
				elif not is_period(line):
					labels[ wav_name ].append( chi2en[ line ] )
				idx += 1
	with open('labels.pkl','w') as f:
		pickle.dump(labels,f)

	import pdb; pdb.set_trace()		
	
