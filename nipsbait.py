import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.preprocessing import sequence
import string
import re

MATCH_MULTIPLE_SPACES = re.compile("\ {2,}")
SEQUENCE_LENGTH = 20

UNK = "<UNK>"
PAD = "<PAD>"

vocab = open("clickbait-detector/data/vocabulary.txt").read().split("\n")
invvocab = dict((word, i) for i, word in enumerate(vocab))
model_path = "clickbait-detector/models/detector.h5"
model = load_model(model_path)


def text_to_indices(text):
    text = text.replace(':', '')
    for punctuation in string.punctuation:
        text = text.replace(punctuation, " " + punctuation + " ")
    for i in range(10):
        text = text.replace(str(i), " " + str(i) + " ")
    text = MATCH_MULTIPLE_SPACES.sub(" ", text)
    return [invvocab.get(word, invvocab[UNK]) for word in text.lower().split()]


def predict(headlines):
    iheadlines = texts_to_indices(headlines)
    inputs = sequence.pad_sequences(iheadlines, maxlen=SEQUENCE_LENGTH)
    clickbaitiness = model.predict(inputs)[:, 0]
    return clickbaitiness


def fracunks(texts):
    iunk = invvocab[UNK]
    nunks = [inds.count(iunk) for inds in texts_to_indices(texts)]
    ntokens = [len(inds) for inds in texts_to_indices(texts)]
    return nunks, ntokens, [a/b for a, b in zip(nunks, ntokens)]


def texts_to_indices(texts):
    return [text_to_indices(text) for text in texts]


def nipsbait():
    nips = pd.read_json('nips.json/nips.json')
    nips['clickbaitiness'] = predict(nips['title'])
    nips['nunks'], nips['ntokens'], nips['fracunks'] = fracunks(nips['title'])
    return nips


def plot_nipsbait(df):
    fig, ax = plt.subplots()
    ax = df['clickbaitiness'].groupby(df['year']).mean().plot()
    ax.legend()
    fig.savefig("nipsbait.png", bbox_inches='tight')
