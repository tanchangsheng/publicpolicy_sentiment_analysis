import argparse # parse arguments as a dictionary for easy access
import gzip # to open gzip compressed files
import pickle # serialize python object(vector and classifier)
import time # to keep track of time taken to run vertain segments of code

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics # show performance metrics of classifier
from sklearn.model_selection import train_test_split # split testing and training data

# basic preprocessing
from features import preprocessor as pp

# magic
import autosklearn.classification



parser = argparse.ArgumentParser(description='Parameters for automl')

# input data
parser.add_argument('--data', type=str, required=True, help='Location of the labelled training data')
parser.add_argument('--pipeline', type=str, required=True, help='Name of pipeline produced')
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

	# convert list of rows as string into list of list of words. Each list of words represent words in a row
	# X = [x.split() for x in X]

	print('Transforming list of documents to sparse matrix')
	vtr = TfidfVectorizer()
	# converts each list of words in list of list of words to sparse matrix
	X = vtr.fit_transform(X)
	print('X (sparse matrix) shape: %s' % str(X.shape))

	# split data for 
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

	automl = autosklearn.classification.AutoSklearnClassifier(
        # time_left_for_this_task=720, per_run_time_limit=30,
        # include_estimators=["LogisticRegression","SVC", "LinearSVC", "NuSVC", "GaussianNB", "MultinomialNB", "BernoulliNB","KNeighborsClassifier","AdaBoostClassifier","RandomForestClassifier"],
        tmp_folder='/tmp/autosklearn_cv_tmp',
        output_folder='/tmp/autosklearn_cv_out',
        delete_tmp_folder_after_terminate=False,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 10})

	t0 = time.time()
	print ('X_train shape: %s' % str(X_train.shape))
	print ('X_test shape: %s' % str(X_test.shape))
	print ('Start of cross validation fit')
    # fit() changes the data in place, but refit needs the original data. We
    # therefore copy the data. In practice, one should reload the data
	automl.fit(X_train.copy(), y_train.copy(), dataset_name='Emotion')

	m, s = divmod(time.time() - t0, 60)
	h, m = divmod(m, 60)
	print('End of cross validated fitting. Time taken %d:%02d:%.2f (hh:mm:ss)' % (h,m,s))
	print('Start of autofit.')
	t0 = time.time()
	# During fit(), models are fit on individual cross-validation folds. To use
    # all available data, we call refit() which trains all models in the
    # final ensemble on the whole dataset.
	automl.refit(X_train.copy(), y_train.copy())
	m, s = divmod(time.time() - t0, 60)
	h, m = divmod(m, 60)
	print('Autofit completed in %d:%02d:%.2f (hh:mm:ss)' % (h,m,s))


	

	t0 = time.time()
	
	pipeline = cmd_args['pipeline']
	if pipeline:
		print('Serializing the classifier to %s' % pipeline)
		pickle.dump(automl, gzip.open(pipeline, 'wb'))
		print('Done in %.2f sec' % (time.time()-t0))

	print(automl.show_models())

	predictions = automl.predict(X_test)
	print("Accuracy score", metrics.accuracy_score(y_test, predictions))

	cm = metrics.confusion_matrix(y_test, predictions)
	micro = metrics.f1_score(y_test, predictions, average='micro')
	macro = metrics.f1_score(y_test, predictions, average='macro')
	weighted = metrics.f1_score(y_test, predictions, average='weighted')
	if automl.coef_.shape[0] == 1:
		binary = metrics.f1_score(y_test, predictions, average='binary')

	print('confusion matrix:\n\n%s\n\n' % cm)
	print('micro f1: %s\n' % micro)
	print('macro f1: %s\n' % macro)
	print('weighted f1: %s\n' % weighted)
	if binary:
		print('binary f1: %s\n' % binary)
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
