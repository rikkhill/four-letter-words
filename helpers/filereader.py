"""Helpers for reading in files."""


def get_unix_dict():

    words = []

    with open("./data/unix_dictionary.txt") as f:
        words = f.readlines()

    words = [w.strip() for w in words]

    return words


def get_scrabble_dict():

    words = []

    with open("./data/scrabble_dictionary.txt") as f:
        words = f.readlines()

    words = [w.strip() for w in words]

    return words

