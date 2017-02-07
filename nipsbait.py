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


def words_to_indices(words):
    return [invvocab.get(word, invvocab[UNK]) for word in words]


def clean(text):
    text = text.replace(':', '')
    for punctuation in string.punctuation:
        text = text.replace(punctuation, " " + punctuation + " ")
    for i in range(10):
        text = text.replace(str(i), " " + str(i) + " ")
    text = MATCH_MULTIPLE_SPACES.sub(" ", text)
    return text


def predict(headlines):
    headlines = [clean(headline).lower().split() for headline in headlines]
    iheadlines = [words_to_indices(headline) for headline in headlines]
    inputs = sequence.pad_sequences(iheadlines, maxlen=SEQUENCE_LENGTH)
    clickbaitiness = model.predict(inputs)[:, 0]
    return clickbaitiness


def nipsbait():
    nips = pd.read_json('nips.json/nips.json')
    nips['clickbaitiness'] = predict(nips['title'])
    return nips


def plot_nipsbait():
    nips = nipsbait()
    fig, ax = plt.subplots()
    ax = nips['clickbaitiness'].groupby(nips['year']).mean().plot()
    ax.legend()
    fig.savefig("nipsbait.png", bbox_inches='tight')
