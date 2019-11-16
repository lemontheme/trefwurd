import csv
import gzip
import html
import logging
import unicodedata
from collections import defaultdict
import re
from math import log2
import operator as op
import itertools as it
from pathlib import Path
from typing import Sequence, Any, Union
import email.utils
import datetime

logger = logging.getLogger(__name__)


def create_file(path: Path, contents=None):
    path.touch()
    if contents:
        path.open("w", encoding="utf-8").write(contents)


def create_dir(path: Path):
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory.")
    if not path.exists():
        path.mkdir()


def load_model_package(path: Union[Path, str]):
    pass


def package_lemmatizer_data(lemmatizer, ):
    pass


def load_extra_lang_data(lang_code: str):
    pass



def get_utc_time_as_iso():
    return (datetime.datetime
            .now()
            .replace(microsecond=0)
            .isoformat())


def generate_meta(existing_meta):
    return {
        "lang": existing_meta.get("lang", None),
        "time_created": get_utc_time_as_iso()
    }


def get_folds(data, n_splits=5):
    n_samples = len(data)
    split_idxs = it.chain(
        (n_samples // n_splits + 1 for _ in range(n_samples % n_splits)),
        it.repeat(n_samples // n_splits),
    )
    folds = []
    while data:
        split_idx = next(split_idxs)
        fold, data = data[:split_idx], data[split_idx:]
        folds.append(fold)
    return folds


def argmax_i(array: Sequence[Any]):
    return max(range(len(array)), key=array.__getitem__)


def argmin_i(array: Sequence[Any]):
    return min(range(len(array)), key=array.__getitem__)


def smoothed_probs(counts, alpha=1e-8):
    dnm = sum(c for c in counts) + alpha * len(counts)
    return tuple((c + alpha) / dnm for c in counts)


def binary_entropy(counts, class_weights=(1, 1)) -> float:
    pos, neg = counts
    if not (pos and neg):
        return 0.0
    pos = pos * class_weights[0]
    neg = neg * class_weights[1]
    p_pos = pos / (pos + neg)
    p_neg = neg / (pos + neg)
    return -(p_pos * log2(p_pos) + p_neg * log2(p_neg))


# def modified_binary_entropy(
#     p_pos: float,  # smoothed
#     p_neg: float,  # smoothed
#     neg_penalty: float = 1.0,
#     class_weights=(1, 1),
#     *args,
#     **kwargs,
# ):
#     # fmt: off
#     p_pos *= class_weights[0]
#     p_neg *= class_weights[1]
#
#     return (
#             - (
#                     (p_pos * log2(p_pos)) + (p_neg * log2(p_neg))
#             )
#             + (
#                     (p_neg - p_pos) / (p_neg + p_pos) * max(neg_penalty, 0)  # penalty value can't be negative.
#             )
#     )
#     # fmt: on

# def monkeypatch_spacy_nl(lemmatizer: SpacyLemmatizer):
#     try:
#         from spacy.lang.nl import DutchDefaults  # ignore import warnings
#     except ImportError:
#         raise RuntimeError("Couldn't monkeypatch spaCy. Check if it's installed.")
#     else:
#
#         def create_lemmatizer(cls, nlp=None):
#             return lemmatizer
#
#         DutchDefaults.create_lemmatizer = create_lemmatizer.__get__(DutchDefaults, None)
#
#
# def read_lexical_data(f, min_freq=0):
#     lex_data = defaultdict(dict)
#     with open(f) as fp:
#         for line in fp:
#             tok, lem, pos, cgn_pos, freq = line.split("\t")
#             if int(freq) >= min_freq:
#                 tok = tok.strip()
#                 lem = lem.strip()
#                 lex_data[pos.lower()][tok] = lem
#     logger.debug("{} items read from {}".format(sum(len(v) for v in lex_data.values()), f))
#     return lex_data
#
#
# def character_ngrams(string, min_gram=2, max_gram=3):
#     string_length = len(string)
#     min_gram, max_gram = map(lambda n: min(n, string_length), (min_gram, max_gram))
#     max_gram += 1
#     for start_idx in range(string_length):
#         for end_idx in range(start_idx + min_gram, start_idx + max_gram):
#             if end_idx > string_length:
#                 break
#             else:
#                 yield string[start_idx:end_idx]
