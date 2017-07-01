from helpers import filereader, text
from keras.models import load_model
import numpy as np
import random
import pandas as pd


# Get dictionary words
# Load in all words
words = filereader.get_unix_dict() + filereader.get_scrabble_dict()
words = [w.lower() for w in words]
words = [text.romanify(w) for w in words]
words = list(set(words))
vectors = [text.vectorise(w) for w in words]
random.shuffle(vectors)
vectors = np.asarray(vectors)

# Generate four-letter combinations
fakewords = [text.random_word() for _ in range(15000)]

# Make sure fake words are not real words
fakewords = [w for w in fakewords if w not in words]
fakewords = list(set(fakewords))
fakewords = fakewords[:7000]
fake_vectors = [text.vectorise(w) for w in fakewords]
fake_vectors = np.asarray(fake_vectors)

# Load the model

model = load_model('./models/flw_auto.h5')

word_scores = {}

fakeword_scores = {}

c = 0
for w in words:
    c += 1
    if c % 1000 == 0:
        print(c)
    word_vector = np.asarray([text.vectorise(w)])
    word_scores[w] = model.evaluate(word_vector, word_vector, 1, verbose=0)

c = 0
for f in fakewords:
    c += 1
    if c % 1000 == 0:
        print(c)
    word_vector = np.asarray([text.vectorise(f)])
    fakeword_scores[f] = model.evaluate(word_vector, word_vector, 1, verbose=0)

realwords = pd.DataFrame(list(word_scores.items()))
fakewords = pd.DataFrame(list(fakeword_scores.items()))

realwords.to_csv("./data/realword_scores.csv")
fakewords.to_csv("./data/fakeword_scores.csv")

