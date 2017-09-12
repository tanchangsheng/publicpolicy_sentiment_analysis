import sys
import json
import gzip
import logging
import time
import operator
import re
import numpy as np
import nltk
from collections import Counter


from features import preprocessor as pp

# Function:
# input
# dictionary of features to extract and the corresponding directory path to relevant files
# output parameters
# list of feature-value mappings : [{"positive_help":2.134},...]
def extract(data, f_dict, preprocess, stop_words):
    """ Extract features from list of data as list of dictionary like features

    """
    if not data or not isinstance(data, list):
        raise Exception('Invalid data, expecting a list')

    if not f_dict or not isinstance(f_dict, dict):
        raise Exception('Invalid features dict, expecting a dict')

    fdict_list = []
    # iterate through each row and obtain features
    for row in data:

        row_fdict =  {}

        if "emoticons" in f_dict:
            row_fdict.update(extract_emoticons(row, f_dict["emoticons"]))

        if "word_feature" in f_dict:
            row_fdict.update(extract_words(row, f_dict["word_feature"]))

        if "pos" in f_dict:
            row_fdict.update(extract_pos(row))

        row_fdict.update(extract_punctuations(row))

        row_fdict.update(extract_uppercase_stats(row))

        if preprocess:
            if stop_words:
                row = pp.process_data(row, stop_words)
            else:
                row = pp.process_data(row, None)
        # if preprocess:
        #     if stop_words:
        #         data = [pp.process_data(row, stop_words) for row in data]
        #     else:
        #         data = [pp.process_data(row, None) for row in data]

        if "fiveWoneH" in f_dict:
            row_fdict.update(extract_5w1h(row, f_dict["fiveWoneH"]))

        if "ngrams" in f_dict:
            row_fdict.update(extract_ngrams(row, (int(f_dict["ngrams"][0]),int(f_dict["ngrams"][1]))))

        if "emotion_words" in f_dict:
            row_fdict.update(extract_emotion_words(row, f_dict["emotion_words"]))

        if "lsd_words" in f_dict:
            row_fdict.update(extract_lsd_words(row, f_dict["lsd_words"]))

        row_fdict.update(extract_interrogative_sentences(row))

        row_fdict.update(extract_sentence_stats(row))

        # analyze_repeated_chars(row, num_repeat=3, include_numeric=False, output_single_char=True)

        # appends dictionary to list of dict features
        fdict_list.append(row_fdict)

    return fdict_list


def extract_emotion_words(row, emotion_words):
    ''' Analyze emotion words
		    Args
			row: A row of data.
			emotion_words: dictionary of emotion:word pair
	'''
    emo_feat = dict()
    if row and len(row.strip()) > 0:
        #row = re.sub(r'[0-9!@#$%^&*+=()\[\]\?<>/\\-_\.,;:"\']', ' ', row)
        # word_arr = row.lower().split()
        word_arr = row.split()
        for k, v in emotion_words.items():
            emo_feat[k] = len([i for i in word_arr if i in v])

        emo_feat['emo_words_count'] = np.sum(list(emo_feat.values()))
        return emo_feat

# lenient version
# def extract_lsd_words(row, lsd_words):
#     ''' Analyze emotion words
# 		    Args
# 			row: A row of data.
# 			lsd_words: dictionary of dictionary of political sentiment words
# 	'''
#     lsd_feat = dict()
#     if row and len(row.strip()) > 0:
#         row = row.lower()
#         row = re.sub(r'[0-9!@#$%^&*+=()\[\]\?<>/\\-_\.,;:"\']', ' ', row)
#         counts = []
#
#         for k, v in lsd_words.items():
#             matched = []
#             for k2, v2 in v.items():
#                 m = [i for i in row.split() if i in v2]
#                 matched += m
#
#             sent = dict()
#             sent['type'] = k
#             sent['words'] = [i[0] for i in Counter(matched).most_common()]
#             sent['counts'] = [i[1] for i in Counter(matched).most_common()]
#             if len(matched) > 0:
#                 counts.append(sent)
#
#         for c in counts:
#             # e.g. {"positive":4}
#             lsd_feat['lsd_' + c['type']] = sum(c['counts'])
#
#     return lsd_feat

# strict version
def extract_lsd_words(row, lsd_words):
    '''
    Analyze emotion words
    Args
    row: A row of data.
    lsd_words: dictionary of dictionary of political sentiment words
	'''
    lsd_feat = dict()
    if row and len(row.strip()) > 0:
        # row = row.lower()
        # row = re.sub(r'[0-9!@#$%^&*+=()\[\]\?<>/\\-_\.,;:"\']', ' ', row)
        counts = []

        for k, v in lsd_words.items():

            for k2, v2 in v.items():
                matched = [i for i in row.split() if i in v2]

                sent = dict()
                sent['type'] = k + "_" + k2
                sent['words'] = [i[0] for i in Counter(matched).most_common()]
                sent['counts'] = [i[1] for i in Counter(matched).most_common()]
                if len(matched) > 0:
                    counts.append(sent)

        for c in counts:
            # e.g. {"positive_help":2}
            for i, w in enumerate(c['words']):
                lsd_feat['lsd_' + c['type'] + '_' + w] = c['counts'][i]
        return lsd_feat



def extract_emoticons(row, emoticons):
    ''' Analyze emotion words
	Args
		row: A row of data.
		emoticons: list of emoticons
	'''
    emo_feat = dict()

    if row and len(row.strip()) > 0:
        for i in emoticons:
            count = row.count(i)
            if count > 0:
                emo_feat['emoticon_' + i] = count
        return emo_feat

def extract_words(row, params):
    """ Extract words as features
		Args
            row: A row of data
			stop_words: List of stop words
			case_insensitive: Use case insensitive?
			strip_entities: Strip entities (hashtags, user mentions, URLs)?
			strip_punctuation: Strip punctuations?
			strip_numbers: Strip numeric strings?
			strip_repeated_chars: Strip repeated characters?
			min_token_length: Minimum word token length
			max_features: Maximum number of feature tokens
    """
    word_feat = dict()
    if row and len(row.strip()) > 0:

        if params["case_insensitive"]:
            row = row.lower()

        if params["strip_entities"]:
            row = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", row).split())

        if params["strip_punctuation"]:
            row = re.sub('[^a-zA-Z0-9]', ' ', row)

        if params["strip_numbers"]:
            row = re.sub('[0-9]', ' ', row)

        if params["strip_repeated_chars"]:
            row = re.sub(r'((\w)\2{2,})', ' ', row)

        if params["stop_words"]:
            row = row.replace('_', ' ').replace('-', ' ')
            x = [i.lower() for i in row.split() if i.lower() not in params["stop_words"]]
            row = ' '.join(x)

        for t in [x for x in row.split() if len(x) >= params["min_token_length"]]:
            if 'word_' + t not in word_feat:
                word_feat['word_' + t] = 0
            word_feat['word_' + t] += 1

        if params["max_features"]:
            # sorts dictionary of word features in descending order of occurence
            word_feat = sorted(word_feat.items(), key=operator.itemgetter(1), reverse=True)
            # creates dictionary of features for top max_features number of features
            word_feat = dict(word_feat[:params["max_features"]])

        return word_feat

def _extract_ngrams(row, n):
    row = row.split()
    ngram_list = []
    for i in range(len(row) - n):
        ngram = ''
        for j in range(i, i + n):
            ngram += row[j] + ' '
        ngram_list.append(ngram.strip())

    return ngram_list

def extract_ngrams(row, ngram_range):
    ''' Analyze word ngrams
    Args
        row: A row of data
        ngram_range: Minimum and maximum n-gram sizes
    '''
    dict_word_ngram = dict()
    if row and len(row.strip()) > 0:

        row = row.lower()

        ngrams = []
        for i in range(ngram_range[0], ngram_range[1]+1):
            ngrams += _extract_ngrams(row, i)

        for ngram in ngrams:
            if ngram not in dict_word_ngram:
                dict_word_ngram[ngram] = 0
            dict_word_ngram[ngram] += 1

        return dict_word_ngram

def word_count(row, keyword):

    count = 0

    for word in row:
        if word == keyword:
            count += 1
    return count

# def extract_5w1h(row, param):
#
#     dict_5w1h = dict()
#     if row and len(row.strip()) > 0:
#
#         param = (int(param[0]),int(param[1]),int(param[2]),int(param[3]),int(param[4]),int(param[5]))
#
#         row = row.lower()
#         row = re.sub(r'[0-9!@#$%^&*+=()\[\]\?<>/\\-_\.,;:"\']', ' ', row)
#         row = row.split()
#
#         totalcount = 0
#         count = 0 # number of type of variables
#
#         if param[0] == 1:
#             totalcount += word_count(row, 'who')
#         if param[1] == 1:
#             totalcount += word_count(row, 'what')
#         if param[2] == 1:
#             totalcount += word_count(row, 'when')
#         if param[3] == 1:
#             totalcount += word_count(row, 'where')
#         if param[4] == 1:
#             totalcount += word_count(row, 'why')
#         if param[5] == 1:
#             totalcount += word_count(row, 'how')
#
#         avg = totalcount * 1.0 / len(row)
#
#         dict_5w1h['dict_5w1h_avg'] = avg
#
#         return dict_5w1h

def extract_5w1h(row, param):

    dict_5w1h = dict()
    if row and len(row.strip()) > 0:

        param = (int(param[0]),int(param[1]),int(param[2]),int(param[3]),int(param[4]),int(param[5]))

        row = row.lower()
        row = re.sub(r'[0-9!@#$%^&*+=()\[\]\?<>/\\-_\.,;:"\']', ' ', row)
        row = row.split()

        totalcount = 0
        count = 0 # number of type of variables
        num_words = len(row)

        if param[0] == 1:
            this_count = word_count(row, 'who')
            dict_5w1h['who'] = this_count / num_words
            totalcount += this_count
        if param[1] == 1:
            this_count = word_count(row, 'what')
            dict_5w1h['what'] = this_count / num_words
            totalcount += this_count
        if param[2] == 1:
            this_count = word_count(row, 'when')
            dict_5w1h['when'] = this_count / num_words
            totalcount += this_count
        if param[3] == 1:
            this_count = word_count(row, 'where')
            dict_5w1h['where'] = this_count / num_words
            totalcount += this_count
        if param[4] == 1:
            this_count = word_count(row, 'why')
            dict_5w1h['why'] = this_count / num_words
            totalcount += this_count
        if param[5] == 1:
            this_count = word_count(row, 'how')
            dict_5w1h['how'] = this_count / num_words
            totalcount += this_count

        avg = totalcount * 1.0 / num_words

        dict_5w1h['dict_5w1h_avg'] = avg

        return dict_5w1h


def extract_pos(row):

    dict_pos = dict()

    if row and len(row.strip()) > 0:

        sents = nltk.sent_tokenize(row)

        for sent in sents:
            sent_pos_dict = dict()
            sent = nltk.word_tokenize(sent)
            sent_pos = nltk.pos_tag(sent)
            # gets all the pos and their counts
            for i in sent_pos:
                if i[1] not in sent_pos_dict:
                    sent_pos_dict[i[1]] = 1
                else:
                    sent_pos_dict[i[1]] += 1
            # for each pos, append the mean occurence in sentence
            for k in sent_pos_dict:
                if k not in dict_pos:
                    dict_pos[k] = [sent_pos_dict[k] / len(sent)]
                else:
                    dict_pos[k].append(sent_pos_dict[k] / len(sent))
        # mean occurence of each pos in text.
        for k in dict_pos:
            dict_pos[k] = np.mean(dict_pos[k])


        return dict_pos

def extract_interrogative_sentences(texts):
    ''' Analyze interrogative sentences
        Args
        texts: Input text
    '''
    iwords = ['who', 'what', 'when', 'where', 'why', 'whose', 'whom', 'how']

    if texts and len(texts.strip()) > 0:
        sent_feat = dict()
        sentences = nltk.sent_tokenize(texts)
        num_sent = len(sentences)
        num_interrogative_sents = 0
        # one list item represents num of int word in one sentence2
        num_int_words_per_sent = []

        iword_count = [0 for i in iwords]

        for sent in sentences:
            num_int_words = 0
            for i, qw in enumerate(iwords):
                sent = sent.lower()
                qw_count = sent.count(qw)
                num_int_words += qw_count
                iword_count[i] += qw_count # adds to count at pos of iword

            num_int_words_per_sent.append(num_int_words)
            if num_int_words > 0:
                num_interrogative_sents += 1


        sent_feat['num_interrogative_sents'] = num_interrogative_sents
        sent_feat['pct_interrogative_sents'] = float('%.4f' % (float(num_interrogative_sents * 1.0 /num_sent)))

        for i, qw in enumerate(iwords):
            qw_count = iword_count[i]
            sent_feat['num_'+qw + '_sents'] = qw_count
            sent_feat['pct_'+qw + '_sents'] = float('%.4f' % (float(qw_count * 1.0/num_sent)))

        return sent_feat


def extract_uppercase_stats(texts):
    '''
    Analyze capitalized words
    Args
    texts: Input text
    '''
    if texts and len(texts.strip()) > 0:
        cap_feat = dict()
        sent_counts = []
        for sent in nltk.sent_tokenize(texts):
            sent = re.sub(r'[^\w\s]', '', sent) # remove special char
            sent = re.sub(r'[0-9a-z]', '', sent) #remove numbers and lowercase letter
            sent = "".join(sent.split())
            sent_counts.append(len(sent))

            cap_feat['mean_cap_words_per_sentence'] = float('%.4f' % np.mean(sent_counts))
            cap_feat['max_cap_words_per_sentence'] = np.max(sent_counts)

            return cap_feat

def extract_sentence_stats(texts):
    '''
    Analyze sentence length
    Args
    texts: Input text
    '''
    if texts and len(texts.strip()) > 0:
        sent_feat = dict()
        sent_lens = []
        for sent in nltk.sent_tokenize(texts):
            sent_lens.append(len(sent))

            sent_feat['max_length'] = np.max(sent_lens)
            sent_feat['min_length'] = np.min(sent_lens)
            sent_feat['mean_length'] = float('%.4f' % np.mean(sent_lens))
            sent_feat['std_length'] = float('%.4f' % np.std(sent_lens))

            return sent_feat

def extract_punctuations(texts):
    '''
    Analyze punctuation usages, except comma, colon, comma, semi-colon, and full-stop
    Args
    texts: Input text
    '''
    if texts and len(texts.strip()) > 0:
        punc_feat = dict()
        # puncs = re.sub('[a-zA-Z0-9/s,:;\.]', ' ', texts)
        # removes all words and numbers
        puncs = re.sub('[a-zA-Z0-9]', ' ', texts)
        # splits into list of punctuations
        puncs = puncs.split()
        for punc in puncs:
            if punc not in punc_feat:
                punc_feat['num_punc_' + punc] = 0
            punc_feat['num_punc_' + punc] += 1

        punc_feat_mean = dict()
        for punc in punc_feat:
            punc_feat_mean['mean_' + punc] = punc_feat[punc] * 1.0 / len(nltk.word_tokenize(texts))
        punc_feat.update(punc_feat_mean)
        return punc_feat

def analyze_repeated_chars(texts, num_repeat, include_numeric, output_single_char):
    '''
    Analyze repeated characters
    Args
    texts: Input text
    num_repeat: Number of repeated characters
    include_numeric: Include numeric string in the input text
    output_single_char: Only include a single character from the repeated patterns?
    '''
    if texts and len(texts.strip()) > 0:
        texts = texts.lower()
        if not include_numeric:
            pattern = re.compile(r'[0-9]')
            texts = pattern.sub('', texts)
        pattern = re.compile(r'((\w)\2{'+str(num_repeat-1)+',})', re.DOTALL)
        if output_single_char:
            chars = [m[1] for m in pattern.findall(texts)]
        else:
            chars = [m[0] for m in pattern.findall(texts)] # this will give xxxxx, yyy, zzzzzz
        rc_feat = dict()
        for i in chars:
            rc_feat[i] = texts.count(i)

        return rc_feat

# output
# one list of features in dictionary form. One list item contains all features of that row of data
def merge_features(feature_lists):

    if not feature_lists or len(feature_lists) < 1:
        raise Exception('No list of features to merge.')

    if len(feature_lists) == 1:
        return feature_lists

    comb_feature_list = feature_lists[0]
    for i in range(len(feature_lists)):
        # to prevent index out of range
        if (i + 1) == len(feature_lists):
            break
        next_feature_list = feature_lists[i + 1]
        for x, y in zip(comb_feature_list, next_feature_list):
            x.update(y)
        i += 1

    return comb_feature_list
