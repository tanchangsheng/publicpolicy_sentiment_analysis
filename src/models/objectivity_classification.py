import argparse
import numpy as np
import pickle
import time
import math

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import train_test_split # split testing and training data
# pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline



parser = argparse.ArgumentParser(description='Perform emotionality classification with the following arguments')
parser.add_argument('--data', type=str, required=True, help='Location of the text data to be classified')
parser.add_argument('--label_index', type=int, default=0, help='The column index in which class label is stored.')
parser.add_argument('--pos_index', type=int, default=1, help='The column index in which position is stored.')
parser.add_argument('--text_index', type=int, default=2, help='The column index in which text documents are stored')
parser.add_argument('--model', type=str, default='lr', help='Type of classifier used')
parser.add_argument('--output', type=str, required=True, help='Location of the output file')

args = parser.parse_args()
cmd_args = vars(args)

def predict():

    t0 = time.time()
    # get data and label
    x, y, z = load_corpus(cmd_args['data'], cmd_args['label_index'], cmd_args['pos_index'], cmd_args['text_index'])

    # create pipeline
    pl_steps = []
    vtr = TfidfVectorizer()
    pl_steps.append(('vtr', vtr))

    model = cmd_args['model']
    clf = None
    if model == "lr":
        clf = LogisticRegression()
    elif model == "nb":
        clf = MultinomialNB()
    elif model == "sgd":
        clf = SGDClassifier()
    elif model == "svc":
        clf = SVC()
    else:
        logging.error('Invalid classifier name.')
        exit(1)

    pl_steps.append(('clf',clf))
    pipeline = Pipeline(pl_steps)

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=0)

    # y_pred = pipeline.fit(x, y).predict(x)
    # cm = metrics.confusion_matrix(y, y_pred)
    # f1 = metrics.f1_score(y, y_pred, average='weighted')
    #
    # precision = metrics.precision_score(y, y_pred, average='weighted')
    # recall = metrics.recall_score(y, y_pred, average='weighted')
    #
    # print('confusion matrix:\n\n%s\n' % cm)
    # print('precision: %s' % precision)
    # print('recall: %s' % recall)

    y_prob_dist = pipeline.fit(x, y).predict_proba(x)

    # topic_prob_output(y_prob_dist, x, cmd_args['output'])

    entropy_list = []
    num_topics = len(pipeline.classes_)
    for prob_dist in y_prob_dist:
        entropy_list.append(get_entropy(prob_dist, num_topics))

    final_output(entropy_list, y_prob_dist, z,x , cmd_args['output'])



def get_entropy(topic_prob_dist, num_topics):
    entropy = 0
    for prob in topic_prob_dist:
        ent = prob * math.log(prob, 2)
        entropy += ent
    entropy = entropy / num_topics * -1.0
    return entropy

def final_output(entropy_list, y_prob_dist, z, x, file_dir):

    combined = zip(entropy_list, y_prob_dist, z, x)
    combined = sorted(combined, key=lambda k: k[0])

    previous = (None, None, None)
    with open(file_dir, 'w') as f:
        for entropy, dist, pos, text in combined:
            if text == previous[2]:
                continue
            f.write(str(entropy) + '\t' + str(dist) + '\t' + str(pos) + '\t' + str(text))
            previous = (entropy, dist, text)

    f.close()

# def topic_prob_output(y_prob_dist, x, file_dir):
#
#     combined = zip(y_prob_dist, x)
#
#     with open(file_dir, 'w') as f:
#         for dist, text in combined:
#             f.write(str(dist) + '\t' + str(text))
#     f.close()


def load_corpus(data_file, label_index, pos_index, text_index):

    labels = []
    data = []
    pos = []

    with open(data_file, 'r') as f:
        for line in f:
            line_arr = line.split('\t')
            labels.append(line_arr[label_index])
            data.append(line_arr[text_index])
            pos.append(line_arr[pos_index])
    f.close()

    return [data, labels, pos]




if __name__ == "__main__":
    predict()
