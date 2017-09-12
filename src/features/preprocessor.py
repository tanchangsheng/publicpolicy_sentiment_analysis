'''
Pre-process corpus
'''


import re
import unicodedata
import codecs
from nltk.stem.wordnet import WordNetLemmatizer


def process_data(text, stop_words):

    text = text.lower()

    # for facebook posts
    text = text.replace('\n', ' ')
    text = text.replace('\/', '/')

    # for forum posts
    text = re.sub('\w+wrote', ' ', text) # remove usernamewrote
    text = re.sub('gagt\w*', ' ', text) # remove gagt*
    text = re.sub('[a-zA-Z][0-9]+[a-zA-Z]*', ' ', text) # remove phone model, e.g. n9005

    text = re.sub(r'[@]\w+', '', text) # remove @mentions

    text = re.sub('(?:(?:https?|ftp):\/\/|www\.)[A-Z-a-z0-9+&@#\/%?=~_|!:,.;]*[A-Z-a-z0-9+&@#\/%=~_|]', ' ', text) # remove URLs
    text = re.sub('[0-9]*', '', text) # remove numbers
    text = re.sub(r'\b([0-9]*[apAP][mM])\b', '', text) # remove 1am, 03pm, etc
    text = re.sub('[\W]+', ' ', text)

    # removes all the \u842E chinese words but text becomes bytes object
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
    # convert byte object back to string
    text = text.decode("utf-8")

    words = [w.lower() for w in text.split() if len(w) > 1]

    if stop_words:
        words = [w for w in words if w not in stop_words]

    wnl = WordNetLemmatizer()

    # lemmatize word if the length is greater than 4, if not do nothing to the word
    words = [wnl.lemmatize(w) if len(w) > 4 else w for w in words]

    # combine words back into a single string
    return ' '.join(words)
