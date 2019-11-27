import csv
import json
import time
from collections import defaultdict
import statistics
import itertools as it
import random
import logging

from trefwurd import Lemmatizer

elex_f = "/Users/adriaan/Dropbox/data/word_lemma_pos_elex.txt"
print(elex_f)

logging.basicConfig(level=logging.DEBUG)


def get_data():

    with open(elex_f) as fp:
        reader = csv.reader(fp, delimiter="\t")
        examples_by_pos = defaultdict(dict)
        token_w_same_pos_duplicates = set()
        for token, lemma, ud_pos, cgn_pos, cgn_freq in reader:
            if token in token_w_same_pos_duplicates:
                continue
            elif token in examples_by_pos[ud_pos]:
                del examples_by_pos[ud_pos][token]
                token_w_same_pos_duplicates.add(token)
            else:
                examples_by_pos[ud_pos][token] = lemma

    data = [((tok, pos), lem) for pos, tok2lem in examples_by_pos.items()
            for tok, lem in tok2lem.items()]

    random.seed(42)
    random.shuffle(data)

    return data


if __name__ == "__main__":

    size = 1000
    data = get_data()[:size]
    pos = "NOUN"

    train_test_partition_idx = int(len(data) * 0.75)
    train_data, test_data = data[:train_test_partition_idx], data[train_test_partition_idx:]

    test_data = test_data[:min(len(train_data), len(test_data))]

    x_train, y_train = zip(*train_data)
    x_test, y_test = zip(*test_data)

    lemmatizer = Lemmatizer(lang="nl")
    lemmatizer.fit(x_train, y_train)

    f = "/Users/adriaan/Projects/trefwurd/scripts/tmp/persisted_lemmatizer/"
    lemmatizer.save(f)
    lemmatizer_ = Lemmatizer.load(f)
    lemmatizer_.lemmatize("honden", "NOUN")

