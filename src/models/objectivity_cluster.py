from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from features import preprocessor
import gzip
import argparse
import math

from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline


parser = argparse.ArgumentParser(description='')

# input data
parser.add_argument('--data', type=str, required=True, help='Location of corpus')
parser.add_argument('--pos_index', type=int, default=0, help='The column index in which labels are stored.')
parser.add_argument('--text_index', type=int, default=1, help='The column index in which text documents are stored')

# for manual train and predict labels given single set of parameter
parser.add_argument('--cluster', type=str, default='kmeans', help='Type of clustering algorithm')
parser.add_argument('--c_out', type=str, help='Location to store labelled posts/comments.')
parser.add_argument('--tw_out', type=str, help='Location to store topwords per cluster')
parser.add_argument('--num_clusters', type=int, help='Number of clusters')
parser.add_argument('--num_tw', type=int, help='Number of topwords per cluster')

args = parser.parse_args()
# convert args into dictionary
cmd_args = vars(args)

additional_stopwords = ['would', 'said', 'say', 'year', 'day', 'also', 'first', 'last', 'one', 'two', 'people', 'told', 'new',
'could', 'singapore', 'three', 'may', 'like', 'world', 'since','mr','time','even','make','many','take','still','well',
'get','want','made','go','much','dr','might','sent','get']

def main():
    data, pos = load_corpus(cmd_args['data'], cmd_args['pos_index'], cmd_args['text_index'])

    stop_words = stopwords.words('english')
    stop_words += additional_stopwords

    # preprocess data to remove stopwords, punctuation, special characters, unicode characters
    # returns list of list of words. Each inner list of words represents words in a row
    pp_data = [preprocessor.process_data(row, stop_words) for row in data]

    # represent data as sparse matrix for clustering
    data_tfidf_vector = TfidfVectorizer().fit_transform(pp_data)
    # clustering
    kmeans = KMeans(n_clusters=cmd_args['num_clusters']).fit(data_tfidf_vector)

    # finding top words in each cluster
    labels = list(kmeans.labels_)
    cluster_dict = {}
    for label, text in zip(labels, pp_data):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append([text])

    topwords = cluster_top_words(cluster_dict, cmd_args['num_tw'])
    create_topwords(topwords, cmd_args['tw_out'])

    # create pipeline
    pl_steps = []
    vtr = TfidfVectorizer()
    pl_steps.append(('vtr', vtr))
    pl_steps.append(('clf',LogisticRegression()))
    pipeline = Pipeline(pl_steps)

    cluster_dist = pipeline.fit(pp_data, labels).predict_proba(pp_data)

    num_topics = len(pipeline.classes_)
    entropy_list = []
    for dist in cluster_dist:
        entropy_list.append(get_entropy(dist, num_topics))

    create_output(entropy_list, cluster_dist, pos, data, cmd_args['c_out'])
    print('Num clusters: ' + str(num_topics))
    print('Cluster top words file created: ' + str(cmd_args['tw_out']))
    print('Entropy/Topic_dist/pos/text file created: ' + str(cmd_args['c_out']))

def load_corpus(data_file, pos_index, text_index):

    data = []
    pos= []

    with gzip.open(data_file, 'r') as f:
        for line in f:
            line_arr = line.decode('utf8').split('\t')
            data.append(line_arr[text_index])
            pos.append(line_arr[pos_index])
    f.close()

    return [data, pos]

def doc2vec(text, stop_words, vectorizer):
    pp_text = preprocessor.process_data(text, stop_words)
    return vectorizer.transform(list(text))

def create_output(entropy_list, cluster_dist, positions, data, file_dir):
    combined = zip(entropy_list, cluster_dist, positions, data)
    combined = sorted(combined, key=lambda k: k[0])
    with open(file_dir, 'w') as f:
        for entropy, dist, pos, text in combined:
            f.write(str(entropy) + '\t' + str(dist) + '\t' + str(pos) + '\t' + text)
    f.close()


def get_entropy(topic_prob_dist, num_topics):
    entropy = 0
    for prob in topic_prob_dist:
        ent = prob * math.log(prob, 2)
        entropy += ent
    entropy = entropy * -1.0 / (math.log(num_topics,2))
    return entropy
# def get_entropy(topic_prob_dist, num_topics):
#     entropy = 0
#     for prob in topic_prob_dist:
#         ent = prob * math.log(prob, 2)
#         entropy += ent
#     entropy = entropy / num_topics * -1.0
#     return entropy

def top_words(data, num):
    words = []
    for row in data:
        words += word_tokenize(row[0])
    fdist= FreqDist(words)
    return fdist.most_common(num)

# cluster_top_words is a list of list of topwords from each cluster
def unique_words(cluster_top_words):
    common_words = []
    unique_words = []
    all_words = []
    for top_words in cluster_top_words:
        words = [word[0] for word in top_words]
        # if words not in common wrds
        all_words += words

def cluster_top_words(clusters, top_num):
    cluster_list = [clusters[key] for key in clusters.keys()]
    top_words_list = []
    for cluster in cluster_list:
        top_words_list.append(top_words(cluster, top_num))

    common_words = []
    for words in top_words_list:
        words = [word[0] for word in words]
        for word in words:
            common = True
            for words_1 in top_words_list:
                words_1 = [word_1[0] for word_1 in words_1]
                if word not in words_1:
                    common = False
                    break
            if common:
                common_words.append(word)
    toReturn = []
    for words in top_words_list:
        words = [word[0] for word in words if word[0] not in common_words]
        toReturn.append(words)
    return toReturn

def create_topwords(tw_list, file_dir):
    with open(file_dir, 'w') as f:
        for row in tw_list:
            f.write(str(row) + '\n')
    f.close()

if __name__ == '__main__':
	main()
