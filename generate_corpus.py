import spacy

import re

# Get all four-letter words from the English language model
lang = spacy.load("en")
pattern = re.compile("\^[A-z]{4}\$")
tokens = [x.lower_ for x in lang.vocab.__iter__() if pattern.match(x.lower_)]
# De-duplicate
tokens = list(set(tokens))

print(len(tokens))

print(sorted(tokens))