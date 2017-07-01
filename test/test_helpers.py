import unittest
from helpers import text
import numpy as np

class TextTest(unittest.TestCase):

    def test_romanify(self):

        diacritic = "Zhōnghuá Rénmín Gònghéguó"
        romanised = "Zhonghua Renmin Gongheguo"

        self.assertEqual(romanised, text.romanify(diacritic), "Romanised diacritics should resemble ascii")


    def test_vectorisation(self):

        word_1 = "hell"
        word_2 = "heel"

        letter_h = [0] * 26
        letter_h[7] = 1

        letter_e = [0] * 26
        letter_e[4] = 1

        letter_l = [0] * 26
        letter_l[11] = 1

        vector_1 = letter_h + letter_e + letter_l + letter_l
        vector_2 = letter_h + letter_e + letter_e + letter_l

        self.assertEqual(vector_1, text.vectorise(word_1).tolist(), "Word vector should resemble vectors")
        self.assertEqual(vector_2, text.vectorise(word_2).tolist(), "Word vector should resemble vectors")
