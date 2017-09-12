import argparse
import gzip
import json
import logging
import numpy as np
import pickle
import time

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import metrics

# pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline

# util classes
from features import features_extractor as fx
from features import preprocessor as pp

parser = argparse.ArgumentParser(description='Perform emotionality classification with the following arguments')
parser.add_argument('--data', type=str, required=True, help='Location of the text data to be classified')
parser.add_argument('--id_index', type=int, default=0, help='The column index in which doc ID (e.g., URL) are stored.')
parser.add_argument('--text_index', type=int, default=1, help='The column index in which text documents are stored')
parser.add_argument('--vtr', type=str, required=True, help='Location of the vectorizer')
parser.add_argument('--clf', type=str, required=True, help='Location of the classifier')
parser.add_argument('--output', type=str, required=True, help='Location of the output file')

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

args = parser.parse_args()
cmd_args = vars(args)

def predict():
    for k in cmd_args:
        print (str(k) + " : " + str(cmd_args[k]))

    data = load_corpus(cmd_args['data'], cmd_args['text_index'])

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

    t0 = time.time()
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
        if cmd_args['preprocess']:
            data = [pp.process_data(row, stop_words) for row in data]
        vtr = TfidfVectorizer()
        pl_steps.append(('vtr', vtr))
        features_list = data
        t1 = time.time()
        logging.info('End features extraction of (%s docs) in %.2f sec' % (len(features_list), (t1-t0)))
        # make a countvectorizor
    elif cmd_args['feat'] == 'count':
        if cmd_args['preprocess']:
            data = [pp.process_data(row, stop_words) for row in data]
        vtr = CountVectorizer()
        pl_steps.append(('vtr', vtr))
        features_list = data
        t1 = time.time()
        logging.info('End features extraction of (%s docs) in %.2f sec' % (len(features_list), (t1-t0)))

    # Loading classifier binaries and external resources
    vtr = pickle.load(gzip.open(cmd_args['vtr'], 'rb'))
    clf = pickle.load(gzip.open(cmd_args['clf'], 'rb'))

    pl_steps = []
    pl_steps.append(('vtr', vtr))
    pl_steps.append(('clf', clf))

    pipeline = Pipeline(pl_steps)
    y_pred = pipeline.predict(features_list)

    # cm = metrics.confusion_matrix(y_test, y_pred)
    # micro = metrics.f1_score(y_test, y_pred, average='micro')
    # macro = metrics.f1_score(y_test, y_pred, average='macro')
    # weighted = metrics.f1_score(y_test, y_pred, average='weighted')
    # if clf.coef_.shape[0] == 1:
    #     binary = metrics.f1_score(y_test, y_pred, average='binary')
    # # f1 = metrics.f1_score(y_test, y_pred, average='binary')
    #
    # precision = metrics.precision_score(y_test, y_pred)
    # recall = metrics.recall_score(y_test, y_pred)
    #
    # logging.info('confusion matrix:\n\n%s\n' % cm)
    # logging.info('precision: %s' % precision)
    # logging.info('recall: %s' % recall)
    # logging.info('micro f1: %s' % micro)
    # logging.info('macro f1: %s' % macro)
    # logging.info('weighted f1: %s' % weighted)
    # if binary:
    #     logging.info('binary f1: %s' % binary)

    f = open(cmd_args['output'], 'w')

    for i in zip(y_pred, data):
        # f.write(str(i[0]) +'\t' + i[1] +'\n')
        f.write(str(i[0]) +'\t' + i[1])
    f.close()




def load_corpus(data_file, text_index):

    data = []

    with gzip.open(data_file, 'r') as f:
        for line in f:
            line_arr = line.decode('utf8').split('\t')
            data.append(line_arr[text_index])
    f.close()

    return data


if __name__ == "__main__":
    predict()
