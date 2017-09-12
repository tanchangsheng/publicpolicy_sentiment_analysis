import argparse
import gzip
import numpy as np
import math
import sys
import re

from nltk.corpus import stopwords
import gensim
from sklearn import metrics

from features import preprocessor



parser = argparse.ArgumentParser(description='')

# input data
parser.add_argument('--data', type=str, required=True, help='Location of corpus')
parser.add_argument('--pos_index', type=int, default=0, help='The column index in positions are stored.')
parser.add_argument('--text_index', type=int, default=1, help='The column index in which text documents are stored')

# for manual train, predict and test on single set of parameter
parser.add_argument('--num_topics', type=int, help='Location of corpus')
parser.add_argument('--topics', type=str, help='Location to save topics and its most frequest words')
parser.add_argument('--output', type=str, help='Location to store labelled posts/comments.')


args = parser.parse_args()
# convert args into dictionary
cmd_args = vars(args)

additional_stopwords = ['would', 'said', 'say', 'year', 'day', 'also', 'first', 'last', 'one', 'two', 'people', 'told', 'new',
'could', 'singapore', 'three', 'may', 'like', 'world', 'since','mr','time','even','make','many','take','still','well',
'get','want','made','go','much','dr','might','sent','get']

def main():
    # input corpus for topic modelling
    data, pos = load_corpus(cmd_args['data'], cmd_args['pos_index'], cmd_args['text_index'])
    stop_words = stopwords.words('english')
    stop_words += additional_stopwords

    # preprocess data to remove stopwords, punctuation, special characters, unicode characters
    # returns list of list of words. Each inner list of words represents words in a row
    pp_data = [preprocessor.process_data(row, stop_words).split() for row in data]

    # create dictionary
    data_dictionary = gensim.corpora.Dictionary(pp_data)
    print('original dictionary')
    print(data_dictionary)

    # remove extremes and reduce redundant feature space
    data_dictionary.filter_extremes(no_below=5, no_above=0.2)
    print('dictionary after reducing extremely common/rare features')
    print(data_dictionary)

    data_vec = docs2vecs(pp_data, data_dictionary)

    # create lda model for corpus
    lda_model = gensim.models.LdaModel(corpus=data_vec, id2word=data_dictionary, num_topics=cmd_args['num_topics'])

    # save topic
    num_topics = cmd_args['num_topics']
    topics = lda_model.show_topics(num_topics, 10)
    save_topics(cmd_args['topics'], topics)

    topic_dist = []
    for vec in data_vec:
        topic_dist.append(lda_model.get_document_topics(vec))

    topic_entropy = []
    for dist in topic_dist:
        topic_entropy.append(get_entropy(dist, num_topics))

    print('Top words file created: ' + str(cmd_args['topics']))
    create_output(topic_entropy, topic_dist, pos, data ,cmd_args['output'])
    print('Entropy/Topic_dist/pos/text file created: ' + str(cmd_args['output']))



def load_corpus(data_file, pos_index, text_index):

    data = []
    pos = []

    with gzip.open(data_file, 'r') as f:
        for line in f:
            line_arr = line.decode('utf8').split('\t')
            data.append(line_arr[text_index])
            pos.append(line_arr[pos_index])

    f.close()

    return [data, pos]

def docs2vecs ( docs , dictionary ):
    # docs is a list of documents returned by corpus2docs.
    # dictionary is a gensim.corpora.Dictionary object.
    vecs1 = [dictionary.doc2bow(doc) for doc in docs]
    return vecs1

def save_topics(filename, topics):
    with open(filename, 'w') as f:
        for topic in topics:
            f.write(str(topic[0]) + '\t')
            chunk = topic[1]
            chunk = re.sub('[.*+"0-9]+', '', chunk)
            chunk = re.sub('[\s]+', ',', chunk)
            f.write(str(chunk) + '\n')
    f.close()

def get_entropy(topic_prob_dist, num_topics):
    entropy = 0
    for topic, prob in topic_prob_dist:
        ent = prob * math.log(prob, 2)
        entropy += ent
    entropy = entropy * -1.0 / (math.log(num_topics,2))
    # entropy = entropy / num_topics * -1.0
    return entropy

def corpus_ent_stats(entropy_list):
    entropy_list = np.array(entropy_list)
    stats_dict = dict()
    stats_dict['mean'] = np.average(entropy_list)
    stats_dict['med'] = np.median(entropy_list)
    stats_dict['std'] = np.std(entropy_list)
    stats_dict['var'] = np.var(entropy_list)
    stats_dict['max'] = np.amax(entropy_list)
    stats_dict['min'] = np.amin(entropy_list)
    return stats_dict


def create_output(entropy_list, topic_dist, positions, data, file_dir):
    combined = zip(entropy_list, topic_dist, positions, data)
    combined = sorted(combined, key=lambda k: k[0])
    with open(file_dir, 'w') as f:
        for entropy, dist, pos, text in combined:
            f.write(str(entropy) + '\t' + str(dist) + '\t' + str(pos) + '\t' + text)
    f.close()

if __name__ == '__main__':
	main()
