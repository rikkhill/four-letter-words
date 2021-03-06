from keras.layers import Dropout, Dense
from keras.models import Sequential
import numpy as np
import random

from helpers import filereader, text

##############
# ETL process to get word vectors

# Load in all words
words = filereader.get_unix_dict() + filereader.get_scrabble_dict()

# Case-insensitive
words = [w.lower() for w in words]

# De-diacriticise
words = [text.romanify(w) for w in words]

# Deduplicate
words = list(set(words))

# Vectorise
vectors = [text.vectorise(w) for w in words]

# Shuffle
random.shuffle(vectors)

# Matrixify

# Split into test/train

train = np.asarray(vectors[:int((len(vectors)+1)*.80)])
test = np.asarray(vectors[int(len(vectors) * .80 + 1):])

print(train.shape)
print(test.shape)


##############
# The autoencoder


input_word = Input(shape=(104, ))

encoded = Dense(26, activation='relu')(input_word)
encoded = Dense(12, activation='relu')(encoded)
encoded = Dense(6, activation='relu')(encoded)
decoded = Dense(12, activation='relu')(encoded)
decoded = Dense(26, activation='relu')(decoded)
decoded = Dense(104, activation='sigmoid')(decoded)


autoencoder = Model(input_word, decoded)

#encoder = Model(input_word, encoded)

#encoded_input = Input(shape=(13, ))
#decoder_layer = autoencoder.layers[-1]
#decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(train, train,
                epochs=5000,
                batch_size=16,
                shuffle=True,
                validation_data=(test, test),
                verbose=2)


words = ["crud", "ball", "woob", "grue", "xbhq", "lakg", "lcfw"]


for w in words:
    wordvec = np.asarray([text.vectorise(w)])
    print(w)
    print(autoencoder.evaluate(wordvec, wordvec, 1))


autoencoder.save("./models/flw_auto.h5")