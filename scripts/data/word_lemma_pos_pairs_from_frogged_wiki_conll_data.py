import sys
import csv
import os
import gzip
import json
from collections import Counter
import itertools as it
import re
import unicodedata

import plac
from plac import Annotation
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.utils import parse_conll


def _read_txt_gazetteer(f: str):
    with open(f) as fp:
        g = [l.strip() for l in fp]
        print("Loaded gazetteer from {}. Tokens: {}".format(f, len(g)))
        return g


def initialize_mega_gazetteer():
    gazetteer_dir = "data/gazetteers"
    with open(os.path.join(gazetteer_dir, "meta.json")) as fp:
        gazetteer_list = json.load(fp)["gazetteers"]
    return set(
        it.chain.from_iterable(
            _read_txt_gazetteer(os.path.join(gazetteer_dir, g["name"]))
            for g in gazetteer_list
        ))


def to_ascii(s):
    """https://github.com/scikit-learn/scikit-learn/blob/adc1e590d4dc1e230b49a4c10b4cd7b672bb3d69/sklearn/feature_extraction/text.py#L70"""
    nkfd_form = unicodedata.normalize('NFKD', s)
    return nkfd_form.encode('ASCII', 'ignore').decode('ASCII')


@plac.annotations(
    conll_data_dir=Annotation(),
    out_f=Annotation()
)
def main(conll_data_dir, out_f):

    conll_fields = ("_", "token", "lemma", "udpos", "cgn_pos")

    udpos_corrections = {"FOREIGN": "X"}

    min_freq = 15
    min_length = 4
    relevant_udpos = {"VERB",
                      "NOUN",
                      "PROPN",
                      "ADJ",
                      "ADV",
                      "SYM",
                      "NUM",
                      "ADP",
                      "PRON",
                      "PUNCT",
                      "DET"}

    # gazetteer = initialize_mega_gazetteer()

    out_dir = os.path.dirname(out_f)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    def udpos_correct(pos):
        return udpos_corrections.get(pos) or pos

    token_lemma_pos_pairs_counter = Counter()

    for f in tqdm(os.listdir(conll_data_dir)):
        f = os.path.join(conll_data_dir, f)
        if os.path.splitext(f)[-1] == ".gz":
            open_ = gzip.open
        else:
            open_ = open
        with open_(f, "rt") as fp:
            content = fp.read()
        records = []
        for conll_tokens, _ in parse_conll(content, cols=conll_fields):
            for t in conll_tokens:
                token, lemma, udpos = t["token"].strip(), t["lemma"].strip(), t["udpos"]
                cgn_pos = t["cgn_pos"]
                token_ = to_ascii(token)
                if not re.match(r"^[0-9a-z'&-]+$", token_, flags=re.IGNORECASE):
                    continue
                elif token[0].isupper():
                    continue
                elif len(token) < min_length and udpos != "PUNCT":
                    continue
                if udpos == "ADJ" and cgn_pos.startswith("WW("):
                    continue
                token, lemma = token.lower(), lemma.lower()
                records.append(
                    (token, lemma, udpos_correct(udpos), cgn_pos))
        token_lemma_pos_pairs_counter.update(records)

    token_lemma_mapping_index = {}
    for tlp_pair, freq in token_lemma_pos_pairs_counter.items():
        tok, lem, udpos, cgn_pos = tlp_pair
        old_entry = token_lemma_mapping_index.get((tok, udpos))
        if old_entry:
            if freq > old_entry[2]:
                token_lemma_mapping_index[(tok, udpos)] = (lem, cgn_pos, freq)
        else:
            token_lemma_mapping_index[(tok, udpos)] = (lem, cgn_pos, freq)

    token_lemma_pos_pairs = ((tok, lem, udpos, cgn_pos, freq)
                             for (tok, udpos), (lem, cgn_pos, freq)
                             in token_lemma_mapping_index.items()
                             if freq >= min_freq)

    with open(out_f, "wt") as fp:
        writer = csv.writer(fp, delimiter="\t")
        for el in token_lemma_pos_pairs:
            writer.writerow(el)


if __name__ == '__main__':
    plac.call(main)
