import argparse
import gzip
import numpy as np
import math
import sys

from nltk.corpus import stopwords
import gensim
from sklearn import metrics

from features import preprocessor



parser = argparse.ArgumentParser(description='')

# input data
parser.add_argument('--data', type=str, required=True, help='Location of corpus')
parser.add_argument('--label_index', type=int, default=0, help='The column index in which labels are stored.')
parser.add_argument('--pos_index', type=int, default=0, help='The column index in positions are stored.')
parser.add_argument('--text_index', type=int, default=1, help='The column index in which text documents are stored')

# for manual train and predict labels given single set of parameter
parser.add_argument('--predict', type=bool, help='To label corpus of comments/posts based on entropy threshold')
parser.add_argument('--output', type=str, help='Location to store labelled posts/comments.')
# for manual train, predict and test on single set of parameter
parser.add_argument('--num_topics', type=int, help='Location of corpus')
parser.add_argument('--topics', type=str, help='Location to save topics and its most frequest words')
parser.add_argument('--threshold', type=int, help='Percentile of entropy to be considered objective')

# for automated parameter tuning (train predict test cycles) to determine parameters that give highest f1 measure
parser.add_argument('--automate', type=bool, help='automated parameter tuning')
parser.add_argument('--verbose', type=bool, help='level of information to display while runnning code. True gives more details.')
parser.add_argument('--num_topics_range', type=str, help='range of topic numbers to try')
parser.add_argument('--threshold_range', type=str, help='list of percentiles to iterate')



args = parser.parse_args()
# convert args into dictionary
cmd_args = vars(args)

def run():

    if cmd_args['automate']:
        magic()
    elif cmd_args['predict']:
        predict_only()
    elif cmd_args['clustering']:
        cluster() # notebook
    else:
        manual()

# def cluster():



def predict_only():
    # input corpus for topic modelling
    data, pos = load_corpus(cmd_args['data'], cmd_args['pos_index'], cmd_args['text_index'])
    print (len(data))
    stop_words = stopwords.words('english')
    stop_words += ['would', 'said', 'say', 'year', 'day', 'also', 'first', 'last', 'one', 'two', 'people', 'told', 'new',
    'could', 'singapore', 'three', 'may', 'like', 'world', 'since','mr','time','even','make','many','take','still','well',
    'get','want','made','go','much','dr','might','sent','get']


    # preprocess data to remove stopwords, punctuation, special characters etc
    # returns list of list of words. Each inner list of words represents words in a row
    pp_data = [preprocessor.process_data(row, stop_words).split() for row in data]

    # create dictionary
    data_dictionary = gensim.corpora.Dictionary(pp_data)
    print('original dictionary')
    print(data_dictionary)

    # remove extremes and reduce redundant feature space
    data_dictionary.filter_extremes(no_below=3, no_above=0.1)
    print('dictionary after reducing redundant features')
    print(data_dictionary)

    data_vec = docs2vecs(pp_data, data_dictionary)

    # create lda model for corpus
    lda_model = gensim.models.LdaModel(corpus=data_vec, id2word=data_dictionary, num_topics=cmd_args['num_topics'])

    # save topic
    num_topics = cmd_args['num_topics']
    topics = lda_model.show_topics(num_topics, 20)
    save_topics(cmd_args['topics'], topics)

    # get the list of topic distributions for each row in corpus
    topic_dist = []
    for vec in data_vec:
        topic_dist.append(lda_model.get_document_topics(vec))

    # predicted_labels = get_topic(topic_dist)
    topic_entropy = []
    for dist in topic_dist:
        topic_entropy.append(get_entropy(dist,num_topics))

    create_output2(topic_entropy, topic_dist, pos, data ,cmd_args['output'])
    # # get list of entropy for each row in corpus
    # topic_entropy = []
    # for dist in topic_dist:
    #     topic_entropy.append(get_entropy(dist,num_topics))
    #
    # entropy_list = np.array(topic_entropy)
    # ent_stats = corpus_ent_stats(topic_entropy)
    # for k in ent_stats:
    #     print (str(k) + ' ' + str(ent_stats[k]))
    #
    # threshold_ent = np.percentile(topic_entropy, cmd_args['threshold'])
    # predicted_labels = predict(topic_entropy, threshold_ent)
    #
    #
    # create_output(predicted_labels, topic_dist,topic_entropy, data, cmd_args['output'])
    # print('Successfully labelled corpus and saved results at ' + str(cmd_args['output']))


# num_topics_range is a tuple. (5,10)
# percentiles is a list. [50,60,70]
def magic():
    # input corpus for topic modelling
    data, labels = load_corpus(cmd_args['data'], cmd_args['label_index'], cmd_args['text_index'])
    stop_words = stopwords.words('english')
    stop_words += ['would', 'said', 'say', 'year', 'day', 'also', 'first', 'last', 'one', 'two', 'people', 'told', 'new',
    'could', 'singapore', 'three', 'may', 'like', 'world', 'since','mr','time','even','make','many','take','still','well',
    'get','want','made','go','much','dr']


    # preprocess data to remove stopwords, punctuation, special characters etc
    # returns list of list of words. Each inner list of words represents words in a row
    pp_data = [preprocessor.process_data(row, stop_words).split() for row in data]

    # create dictionary
    data_dictionary = gensim.corpora.Dictionary(pp_data)
    print('original dictionary')
    print(data_dictionary)

    # remove extremes and reduce redundant feature space
    data_dictionary.filter_extremes(no_below=3, no_above=0.1)
    print('dictionary after reducing redundant features')
    print(data_dictionary)

    data_vec = docs2vecs(pp_data, data_dictionary)

    results = []
    num_topics_range = cmd_args['num_topics_range']
    num_topics_range = num_topics_range.split(',')
    thresholds = cmd_args['threshold_range']
    thresholds = thresholds.split(',')
    print ('num_topics_range -> ' + str(num_topics_range))
    print ('thresholds -> ' + str(thresholds))
    count = 0
    for num_topics in range(int(num_topics_range[0]), int(num_topics_range[1])):
        for threshold in thresholds:
            count += 1
            print ("Iteration no.: " + str(count))
            results.append(repeated(data_vec, data_dictionary, labels, int(num_topics), int(threshold)))

    sorted_results = sorted(results, key=lambda k: k['f1'], reverse=True)
    print ('Best result -> ' + str(sorted_results[0]))

def repeated(data_vec, data_dictionary, labels, num_topics, threshold):

    # create lda model for corpus
    lda_model = gensim.models.LdaModel(corpus=data_vec, id2word=data_dictionary, num_topics=num_topics)
    # get the list of topic distributions for each row in corpus
    topic_dist = []
    for vec in data_vec:
        topic_dist.append(lda_model.get_document_topics(vec))

    # get list of entropy for each row in corpus
    topic_entropy = []
    for dist in topic_dist:
        topic_entropy.append(get_entropy(dist,num_topics))

    entropy_list = np.array(topic_entropy)
    ent_stats = corpus_ent_stats(topic_entropy)

    threshold_ent = np.percentile(topic_entropy, threshold)
    predicted_labels = predict(topic_entropy, threshold_ent)

    precision = metrics.precision_score(labels, predicted_labels, pos_label='1')
    recall = metrics.recall_score(labels, predicted_labels, pos_label='1')
    f1 = metrics.f1_score(labels, predicted_labels, pos_label='1', average='binary')

    result = {"num_topics": num_topics, "threshold":threshold, "precision":precision, "recall":recall, "f1":f1}

    if cmd_args['verbose']:
        print (result)

    return result

def manual():
    # input corpus for topic modelling
    data, labels = load_corpus(cmd_args['data'], cmd_args['label_index'], cmd_args['text_index'])
    stop_words = stopwords.words('english')
    stop_words += ['would', 'said', 'say', 'year', 'day', 'also', 'first', 'last', 'one', 'two', 'people', 'told', 'new',
    'could', 'singapore', 'three', 'may', 'like', 'world', 'since','mr','time','even','make','many','take','still','well',
    'get','want','made','go','much','dr']


    # preprocess data to remove stopwords, punctuation, special characters etc
    # returns list of list of words. Each inner list of words represents words in a row
    pp_data = [preprocessor.process_data(row, stop_words).split() for row in data]

    # create dictionary
    data_dictionary = gensim.corpora.Dictionary(pp_data)
    print('original dictionary')
    print(data_dictionary)

    # remove extremes and reduce redundant feature space
    data_dictionary.filter_extremes(no_below=3, no_above=0.1)
    print('dictionary after reducing redundant features')
    print(data_dictionary)

    data_vec = docs2vecs(pp_data, data_dictionary)

    # create lda model for corpus
    lda_model = gensim.models.LdaModel(corpus=data_vec, id2word=data_dictionary, num_topics=cmd_args['num_topics'])

    # save topic
    num_topics = cmd_args['num_topics']
    topics = lda_model.show_topics(num_topics, 20)
    save_topics(cmd_args['topics'], topics)

    # get the list of topic distributions for each row in corpus
    topic_dist = []
    for vec in data_vec:
        topic_dist.append(lda_model.get_document_topics(vec))


    # get list of entropy for each row in corpus
    topic_entropy = []
    for dist in topic_dist:
        topic_entropy.append(get_entropy(dist,num_topics))

    entropy_list = np.array(topic_entropy)
    ent_stats = corpus_ent_stats(topic_entropy)
    for k in ent_stats:
        print (str(k) + ' ' + str(ent_stats[k]))

    threshold_ent = np.percentile(topic_entropy, cmd_args['threshold'])
    predicted_labels = predict(topic_entropy, threshold_ent)

    cm = metrics.confusion_matrix(labels, predicted_labels)
    precision = metrics.precision_score(labels, predicted_labels, pos_label='1')
    recall = metrics.recall_score(labels, predicted_labels, pos_label='1')
    f1 = metrics.f1_score(labels, predicted_labels, pos_label='1', average='binary')

    print('confusion matrix: predicted (horizontal), true (vertical) \n\n%s\n' % cm)
    print('precision: %s' % precision)
    print('recall: %s' % recall)
    print('binary f1: %s' % f1)


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
            f.write(str(topic[0]) + '\t' + str(topic[1]) + '\n')
    f.close()

def get_entropy(topic_prob_dist, num_topics):
    entropy = 0
    for topic, prob in topic_prob_dist:
        ent = prob * math.log(prob, 2)
        entropy += ent
    entropy = entropy / num_topics * -1.0
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

def predict(entropy_list, threshold):
    target = []
    count = 0
    for entropy in entropy_list:
        if entropy >= threshold:
            count += 1
            target.append('1')
        else:
            target.append('0')
    print (str(count) + ' / ' + str(len(entropy_list)) + ' or ' + str(round((count/len(entropy_list)), 2)) + ' labelled as objective.')
    return target

def create_output2(labels, topics, positions, data, file_dir):
    combined = zip(labels, topics, positions, data)
    combined = sorted(combined, key=lambda k: k[0])
    with open(file_dir, 'w') as f:
        for label, topic, pos, text in combined:
            f.write(str(label) + '\t' + str(topic) + '\t' + str(pos) + '\t' + text)
    f.close()

def get_topic(topic_dist):
    topic = []
    for dist in topic_dist:
        dist = sorted(dist, key=lambda k: k[1], reverse=True)
        topic.append(dist[0][0])
    return topic

def create_output(labels, topic_dist, topic_entropy, data, file_dir):

    combined = zip(labels, topic_dist,topic_entropy, data)
    with open(file_dir, 'w') as f:
        for label, dist, ent, text in combined:
            f.write(label + '\t' + str(ent) + '\t' + str(dist) + '\t' + text)
    f.close()

if __name__ == '__main__':
	run()
