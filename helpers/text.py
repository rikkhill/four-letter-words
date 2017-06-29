"""Text functions."""

import unicodedata
import numpy as np


def romanify(text):
    """Convert text with diacritic markers to vanilla Roman/ASCII counterpart."""
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii")


def vectorise(word):
    """Produce a vector representation of a word."""

    # Case insensitive
    word = word.lower()
    letter_indices = [ord(c) - ord("a") for c in word]

    vector = []

    for i in letter_indices:
        letter = [0] * 26
        letter[i] = 1
        vector += letter

    return np.array(vector)


def devectorise(word):

    return word