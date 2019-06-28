import csv
import gzip
import html
import logging
import unicodedata
from collections import defaultdict
import re

from .lemmatizer import SpacyLemmatizer


logger = logging.getLogger(__name__)


def monkeypatch_spacy_nl(lemmatizer: SpacyLemmatizer):
    try:
        from spacy.lang.nl import DutchDefaults  # ignore import warnings
    except ImportError:
        raise RuntimeError("Couldn't monkeypatch spaCy. Check if it's installed.")
    else:
        def create_lemmatizer(cls, nlp=None):
            return lemmatizer
        DutchDefaults.create_lemmatizer = create_lemmatizer.__get__(DutchDefaults, None)


def read_lexical_data(f, min_freq=0):
    lex_data = defaultdict(dict)
    with open(f) as fp:
        for line in fp:
            tok, lem, pos, cgn_pos, freq = line.split("\t")
            if int(freq) >= min_freq:
                tok = tok.strip()
                lem = lem.strip()
                lex_data[pos.lower()][tok] = lem
    logger.debug("{} items read from {}".format(sum(len(v) for v in lex_data.values()), f))
    return lex_data


def character_ngrams(string, min_gram=2, max_gram=3):
    string_length = len(string)
    min_gram, max_gram = map(lambda n: min(n, string_length), (min_gram, max_gram))
    max_gram += 1
    for start_idx in range(string_length):
        for end_idx in range(start_idx + min_gram, start_idx + max_gram):
            if end_idx > string_length:
                break
            else:
                yield string[start_idx:end_idx]

