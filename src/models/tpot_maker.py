import argparse # parse arguments as a dictionary for easy access
import gzip # to open gzip compressed files
import pickle # serialize python object(vector and classifier)
import time # to keep track of time taken to run vertain segments of code
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics # show performance metrics of classifier
from sklearn.model_selection import train_test_split # split testing and training data

# basic preprocessing
from features import preprocessor as pp

# magic
from tpot import TPOTClassifier


parser = argparse.ArgumentParser(description='Parameters for automl')

# input data
parser.add_argument('--data', type=str, required=True, help='Location of the labelled training data')
parser.add_argument('--script', type=str, required=True, help='Name of script to train best model')
parser.add_argument('--label_index', type=int, default=0, help='The column index in which labels are stored.')
parser.add_argument('--text_index', type=int, default=1, help='The column index in which text documents are stored')
parser.add_argument('--preprocess', type=bool, default=True, help='Preprocess or not')

args = parser.parse_args()
cmd_args = vars(args)


def abracadabra():


	t0 = time.time()

	y, X = load_corpus(cmd_args['data'], cmd_args['label_index'],cmd_args['text_index'])

	if len(set(y)) == 2:
		y = [int(i) for i in y]
	
	print('End loading training data (%s docs) in %.2f sec' % (len(X), (time.time()-t0)))

    # preprocess data
	if cmd_args['preprocess']:
		X = [pp.process_data(row, None) for row in X]


	print('Transforming list of documents to sparse matrix')
	vtr = TfidfVectorizer()
	# converts each list of words in list of list of words to sparse matrix
	X = vtr.fit_transform(X)
	print('X (sparse matrix) shape: %s' % str(X.shape))

	# split data for 
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

	automl = TPOTClassifier(cv=10, random_state=42, scoring='f1',verbosity=3)

	t0 = time.time()
	print ('X_train shape: %s' % str(X_train.shape))
	print ('X_test shape: %s' % str(X_test.shape))
	print ('Start of fitting model')
	automl.fit(X_train.toarray(), y_train)


	m, s = divmod(time.time() - t0, 60)
	h, m = divmod(m, 60)
	print('End of fitting. Time taken %d:%02d:%.2f (hh:mm:ss)' % (h,m,s))

	t0 = time.time()
	

	print("F1 score %.3f" % tpot.score(X_test, y_test))


	script = cmd_args['script']
	print('Exporting model script: %s' % script)
	automl.export(script)





# Function: Converts labelled data as list of data and target labels
# input parameters:
# data_file: location of labelled data
# label_index index position of label
# text_index: index position of data
# output parameters
# list of list of labels and list of data: [[labels][data]]
def load_corpus(data_file, label_index, text_index):

	labels = []
	data = []

	with gzip.open(data_file, 'r') as f:
		for line in f:
			line_arr = line.decode('utf8').split('\t')
			labels.append(line_arr[label_index])
			data.append(line_arr[text_index])
	f.close()

	return [labels, data]

if __name__ == '__main__':
	abracadabra()