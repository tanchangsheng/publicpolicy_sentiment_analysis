'''
Script to find out best params for a model
'''
import argparse # parse arguments as a dictionary for easy access
import gzip # to open gzip compressed files
import logging # to track and log intermediate steps as well as errors
import pickle # serialize python object(vector and classifier)
import time # to keep track of time taken to run vertain segments of code
import numpy as np
from nltk.corpus import stopwords

# different types of classifiers used
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer # show performance metrics of classifier

# util classes
from features import features_extractor as fx
from features import preprocessor as pp

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
parser.add_argument('--model', type=str, choices=["lr", "nb", "sgd", "svc", "lsvc"], help='Classifier name, i.e., LogisticRegression, MultinomialNB, SGDClassifier')
parser.add_argument('--features_norm', type=str, choices=['l1', 'l2'], default='l2', help='Peform features normalization')
parser.add_argument('--num_topk_features', type=int, help='Number of top K best features for feature extraction')

# output models
parser.add_argument('--nfolds', type=int, default=10, help='Number of folds for cross validation')
args = parser.parse_args()
# convert args into dictionary
cmd_args = vars(args)

def grid_search_cv():
    for k in cmd_args:
        print (str(k) + " : " + str(cmd_args[k]))

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
        pl_steps.append(('vtr', DictVectorizer()))
        t1 = time.time()
        logging.info('End features extraction of (%s docs) in %.2f sec' % (len(features_list), (t1-t0)))
    elif cmd_args['feat'] == 'tfidf':
        # converts list of docs as tring to list of docs as list of their words
        if cmd_args['preprocess']:
            data = [pp.process_data(row, stop_words) for row in data]
        vtr = TfidfVectorizer(max_df=0.8, sublinear_tf=0.8)
        features_list = vtr.fit_transform(data)
        t1 = time.time()
        logging.info('End features extraction of (%s docs) in %.2f sec' % (features_list.shape, (t1-t0)))
        # make a countvectorizor
    elif cmd_args['feat'] == 'count':
        # converts list of docs as tring to list of docs as list of their words
        if cmd_args['preprocess']:
            data = [pp.process_data(row, stop_words) for row in data]
        vtr = CountVectorizer(max_df=0.8)
        features_list = vtr.fit_transform(data)
        t1 = time.time()
        logging.info('End features extraction of (%s docs) in %.2f sec' % (features_list.shape, (t1-t0)))



    # normalize sparse matrix
    if cmd_args['features_norm']:
        logging.info('Pipeline: Normalzing features with %s' % cmd_args['features_norm'])
        pl_steps.append(('normalizer', Normalizer(norm=cmd_args['features_norm'])))

    # select k best features according to chi2 test scores
    if cmd_args['num_topk_features']:
        logging.info('Pipeline: Selecting %d best features by a chi-square test' % cmd_args['num_topk_features'])
        pl_steps.append(('fsel', SelectKBest(chi2, k=cmd_args['num_topk_features'])))
        # pl_steps.append(('fsel', SelectKBest(chi2, k='all')))


    # create classifier
    if model == "lr":
        params = {'clf__penalty': ['l1','l2'], 'clf__C': range(1,10)}
        clf = LogisticRegression()
    elif model == "nb":
        params = {'clf__alpha': np.arange(0, 0.01, 0.001)}
        clf = MultinomialNB()
    elif model == "sgd":
        params = {'clf__loss': ['log', 'hinge'], 'clf__alpha': [0.001, 0.0001, 0.00001, 0.000001]}
        clf = SGDClassifier(shuffle=True)
    elif model == "svc":
        params = {'clf__C': [1, 10, 100, 1000], 'clf__kernel': ['linear', 'rbf', 'poly']}
        clf = SVC()
    elif model == "lsvc":
        params = {'clf__C': [1, 10, 100, 1000], 'clf__loss': ['hinge', 'squared_hinge']}
        clf = LinearSVC()
    else:
        logging.error('Invalid classifier name.')
        exit(1)


    # print ('features length -> ' + str(len(features_list)))
    print ('feature length of first document -> ' + str(features_list[0].shape))
    # print ('feature length of first document -> ' + str(len(features_list[0])))
    # print ('features -> ' + str(features_list[0]))

    pl_steps.append(('clf', clf))

    t0 = time.time()
    pipeline = Pipeline(pl_steps)
    weighted_f1 = make_scorer(f1_score)
    logging.info('Performing Grid Search CV...')
    gsc = GridSearchCV(pipeline, params, scoring=weighted_f1, cv=cmd_args['nfolds'], verbose=1)
    gsc.fit(features_list, labels)
    t1 = time.time()
    logging.info("End in %.4f sec" % (t1 - t0))
    gscores = gsc.cv_results_['mean_test_score']
    gstds = gsc.cv_results_['std_test_score']
    gparams = gsc.cv_results_['params']
    for i in range(len(gscores)):
        print ("mean: %.5f, std: %.5f, params: %s" % (gscores[i], gstds[i], gparams[i]))
    print ("Best score: %s" % gsc.best_score_)
    print ("Best params: %s" % gsc.best_params_)


    # use best parameters and print out top 50 features

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
    grid_search_cv()
