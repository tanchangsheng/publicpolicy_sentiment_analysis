'''
Script to train a model based on specified input.
'''
import argparse # parse arguments as a dictionary for easy access
import gzip # to open gzip compressed files
import logging # to track and log intermediate steps as well as errors
import pickle # serialize python object(vector and classifier)
import time # to keep track of time taken to run vertain segments of code
import json
import numpy as np
from nltk.corpus import stopwords

from sklearn import metrics # show performance metrics of classifier
from sklearn.model_selection import train_test_split # split testing and training data


# different types of classifiers used
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

# pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

# util classes
from features import features_extractor as fx
from features import preprocessor as pp

# Does basic configuration for the logging system by creating a StreamHandler with
# a default Formatter and adding it to the root logger.
logging.basicConfig(level=logging.DEBUG)


parser = argparse.ArgumentParser(description='Find the best parameters using GridSearchCV')

# input data
parser.add_argument('--data', type=str, required=True, help='Location of the labelled training data')
parser.add_argument('--label_index', type=int, default=0, help='The column index in which labels are stored.')
parser.add_argument('--text_index', type=int, default=1, help='The column index in which text documents are stored')

# preprocess
parser.add_argument('--preprocess', type=bool, default=False, help='Preprocess or not')
parser.add_argument('--stop_words', type=bool, default=False, help='Location of the stop words file')

parser.add_argument('--feat', type=str, default='count', help='Feature types: dict, tfidf, count')

# feature variables
parser.add_argument('--emotion_words', type=str, default=None, help='Location of the emotion words file')
parser.add_argument('--lsd_words', type=str, default=None, help='Location of the LSD sentiment words file')
parser.add_argument('--emoticons', type=str, default=None, help='Location of emoticons file')
parser.add_argument('--word_feature', type=bool, default=False, help='Use individual words as feature')
parser.add_argument('--ngrams', type=tuple, default=None, help='Use ngram as feaature')
parser.add_argument('--fiveWoneH', type=tuple, default=None, help='Use avg count of who what when where why how as feaature')
parser.add_argument('--pos', type=bool, default=False, help='Use avg count of pos tag in sentence as feature')


# model variables
parser.add_argument('--model', type=str, choices=["lr", "nb", "sgd", "svc"], help='Classifier name, i.e., LogisticRegression, MultinomialNB, SGDClassifier')
parser.add_argument('--model_params', type=json.loads, help='Model parameters in JSON format')
parser.add_argument('--features_norm', type=str, choices=['l1', 'l2'], default=None, help='Peform features normalization')
parser.add_argument('--num_topk_features', type=int, help='Number of top K best features for feature extraction')

# output models
parser.add_argument('--vtr', type=str, default=None, help='Save the vectorizer to this location')
parser.add_argument('--clf', type=str, default=None, help='Save the classifier to this location')

args = parser.parse_args()
# convert args into dictionary
cmd_args = vars(args)


def train():
    for k in cmd_args:
        print (str(k) + " : " + str(cmd_args[k]))

    model = cmd_args['model']
    model_params = cmd_args['model_params']
    vtr_file = cmd_args['vtr']
    clf_file = cmd_args['clf']

    logging.info ('Finding the parameters for: data =  %s, clf = %s' % (cmd_args['data'], cmd_args['model']))
    model = cmd_args['model']

    t0 = time.time()

    # load data
    labels, data = load_corpus(cmd_args['data'], cmd_args['label_index'],cmd_args['text_index'])

    if len(set(labels)) == 2:
        labels = [int(i) for i in labels]

    t1 = time.time()
    logging.info('End loading training data (%s docs) in %.2f sec' % (len(data), (t1-t0)))


    # load stop_words file
    stop_words = None
    if cmd_args['stop_words']:
        stop_words = stopwords.words('english')
        # stop_words = [i.strip() for i in open(cmd_args['stop_words']).read().split('\n')]


    # emoticons needs to be extracted before preprocessing
    # list of features to extract
    features_dict = dict()
    if cmd_args['emoticons']:

        # load emoticons from file in a list
        emoticons = [i.strip() for i in open(cmd_args['emoticons'], 'r')]

        features_dict["emoticons"] = emoticons


    # feature extraction
    if cmd_args['emotion_words']:

        # load emotion words as a dictionary where k, v equals emotion:word pair
        emo_words_dict = dict()
        for line in open(cmd_args['emotion_words'], 'r'):
            arr = line.strip().split('\t')
            if arr[0] not in emo_words_dict.keys():
                emo_words_dict[arr[0]] = []
            emo_words_dict[arr[0]].append(arr[1])

        # store in features_dict to be processed by fx
        features_dict["emotion_words"] = emo_words_dict

    if cmd_args['lsd_words']:

        # load lsd words as a dictionary of dictionary of words by its polarity and extent
        lsd_words = dict()
        for line in open(cmd_args['lsd_words'], 'r'):
            arr = line.strip().split('\t')
            sent_label = 'negative' if arr[1] == '0' else 'positive'
            if sent_label not in lsd_words.keys():
                lsd_words[sent_label] = dict()
            if '*' in arr[0]:
                if 'partial' not in lsd_words[sent_label]:
                    lsd_words[sent_label]['partial'] = []
                lsd_words[sent_label]['partial'].append(arr[0].lower().replace('*', ''))
            else:
                if 'exact' not in lsd_words[sent_label]:
                    lsd_words[sent_label]['exact'] = []
                lsd_words[sent_label]['exact'].append(arr[0].lower())

        features_dict["lsd_words"] = lsd_words


    if cmd_args['word_feature']:

        features_dict["word_feature"] = {
            "stop_words":stop_words,
            "case_insensitive":True,
            "strip_entities":True,
            "strip_punctuation":True,
            "strip_numbers":True,
            "strip_repeated_chars":True,
            "min_token_length": 3,
            "max_features":1000
        }

    if cmd_args['ngrams']:

        features_dict['ngrams'] = cmd_args['ngrams']

    if cmd_args['fiveWoneH']:

        features_dict['fiveWoneH'] = cmd_args['fiveWoneH']

    if cmd_args['pos']:

        features_dict['pos'] = cmd_args['pos']



    pl_steps = []
    # merge all features into one list where each list item is a row of data with it's dict of feature value pairs
    if cmd_args['feat'] == 'dict' and len(features_dict) > 0:
        print ('DICT FEATURE EXTRACTION: exracting dict features...')
        for k in features_dict:
            print (str(k) + " : " + str(cmd_args[k]))
        features_list = fx.extract(data, features_dict, cmd_args['preprocess'], stop_words)
        vtr = DictVectorizer()
        pl_steps.append(('vtr', vtr))
        t1 = time.time()
        logging.info('End features extraction of (%s docs) in %.2f sec' % (len(features_list), (t1-t0)))
    elif cmd_args['feat'] == 'tfidf':
        # converts list of docs as tring to list of docs as list of their words
        if cmd_args['preprocess']:
            data = [pp.process_data(row, stop_words) for row in data]
        vtr = TfidfVectorizer(max_df=0.8,sublinear_tf=True)
        pl_steps.append(('vtr', vtr))
        features_list = data
        t1 = time.time()
        logging.info('End features extraction of (%s docs) in %.2f sec' % (len(features_list), (t1-t0)))
        # make a countvectorizor
    elif cmd_args['feat'] == 'count':
        # converts list of docs as tring to list of docs as list of their words
        if cmd_args['preprocess']:
            data = [pp.process_data(row, stop_words) for row in data]
        vtr = CountVectorizer(max_df=0.8)
        pl_steps.append(('vtr', vtr))
        features_list = data
        t1 = time.time()
        logging.info('End features extraction of (%s docs) in %.2f sec' % (len(features_list), (t1-t0)))



    # normalize sparse matrix
    if cmd_args['features_norm']:
        logging.info('Pipeline: Normalzing features with %d' % cmd_args['features_norm'])
        pl_steps.appends(('normalizer', Normalizer(norm=cmd_args['features_norm'])))

    # select k best features according to chi2 test scores
    if cmd_args['num_topk_features']:
        logging.info('Pipeline: Selecting %d best features by a chi-square test' % cmd_args['num_topk_features'])
        pl_steps.append(('fsel', SelectKBest(chi2, k=cmd_args['num_topk_features'])))

    # create classifier
    if model == "lr":
        clf = LogisticRegression(penalty=model_params['penalty'], C=model_params['C'])
    elif model == "nb":
        clf = MultinomialNB(alpha=model_params['alpha'])
    elif model == "sgd":
        clf = SGDClassifier(alpha=model_params['alpha'], loss=model_params['loss'], shuffle=True)
    elif model == "svc":
        clf = SVC(kernel=model_params['kernel'], C=model_params['C'])
    else:
        logging.error('Invalid classifier name.')
        exit(1)

    # append classifier to pipeline
    pl_steps.append(('clf', clf))

    t0 = time.time()
    pipeline = Pipeline(pl_steps)
    pipeline.fit(features_list, labels)
    t1 = time.time()
    score = pipeline.score(features_list, labels)
    logging.info('Fitted the model in %.2f sec, score=%.4f' % ((t1-t0), score))


	# display top-k features for each class label
    if cmd_args['feat'] == 'dict':
        feature_names = vtr.get_feature_names()
    k = 50
    top_k = ''

    # # if more than 1 class label
    # if clf.coef_.shape[0] > 1:
    #     for i in sorted(set(labels)):
    #         topk += [feature_names[x] for x in np.argsort(clf.coef_[int(i)-1])[::-1]][:k]
    # # if only 1 class label
    # else:
    #     topk = [(feature_names[x], float('%.4f' % clf.coef_[0][x])) for x in np.argsort(clf.coef_[0])[::-1] if clf.coef_[0][x] > 0][:k]
    #
    # logging.info('Top-K features:\n%s' % top_k)


    # conduct train test split to find out rough performance of trained model
    data_train, data_test, y_train, y_test = train_test_split(features_list, labels, test_size=0.5, random_state=0)
    y_pred = pipeline.fit(data_train, y_train).predict(data_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    micro = metrics.f1_score(y_test, y_pred, average='micro')
    macro = metrics.f1_score(y_test, y_pred, average='macro')
    weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    if clf.coef_.shape[0] == 1:
        binary = metrics.f1_score(y_test, y_pred, average='binary')
    # f1 = metrics.f1_score(y_test, y_pred, average='binary')

    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)

    logging.info('confusion matrix:\n\n%s\n' % cm)
    logging.info('precision: %s' % precision)
    logging.info('recall: %s' % recall)
    logging.info('micro f1: %s' % micro)
    logging.info('macro f1: %s' % macro)
    logging.info('weighted f1: %s' % weighted)
    if binary:
        logging.info('binary f1: %s' % binary)

    if vtr_file:
        t0 = time.time()
        logging.info('Serializing the vectorizer to %s' % vtr_file)
        pickle.dump(vtr, gzip.open(vtr_file, 'wb'))
        t1 = time.time()
        logging.info('Done in %.2f sec' % (t1-t0))

    if clf_file:
        logging.info('Serializing the classifier to %s' % clf_file)
        pickle.dump(clf, gzip.open(clf_file, 'wb'))
        t1 = time.time()
        logging.info('Done in %.2f sec' % (t1-t0))

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
	train()
