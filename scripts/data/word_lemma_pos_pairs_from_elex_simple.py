import sys
import csv
import os
import importlib.util
import functools as ft
from collections import Counter
import re

import plac
from plac import Annotation

sys.path.append(os.getcwd())

spec = importlib.util.spec_from_file_location("cgn2ud", "data/cgn2ud.py")
cgn2ud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cgn2ud)

cgn2udpos = cgn2ud.cgn2ud

import unicodedata


def to_ascii(s):
    """https://github.com/scikit-learn/scikit-learn/blob/adc1e590d4dc1e230b49a4c10b4cd7b672bb3d69/sklearn/feature_extraction/text.py#L70"""
    nkfd_form = unicodedata.normalize('NFKD', s)
    return nkfd_form.encode('ASCII', 'ignore').decode('ASCII')


@ft.lru_cache(maxsize=500)
def make_cgn_tag(pos_head, pos_morph):
    return "{}({})".format(pos_head, pos_morph)


@plac.annotations(
    preparsed_elex_f=Annotation(),
    out_f=Annotation()
)
def main(preparsed_elex_f, out_f):

    out_dir = os.path.dirname(out_f)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    min_token_length = 1  # set this to 0 to see what comes out.

    relevant_udpos = {"VERB",
                      "NOUN",
                      "PROPN",
                      "ADJ",
                      "PUNCT",
                      "ADV",
                      "NUM",
                      "ADP",
                      "PRON",
                      "DET"}

    token_lemma_mapping_index = {}

    with open(preparsed_elex_f) as fp:
        for line in fp:
            token, lemma, cgn_pos_head, cgn_pos_morph, freq = line.split("\t")
            freq = int(freq)
            token, lemma = token.strip(), lemma.strip()

            cgn_tag = make_cgn_tag(cgn_pos_head, cgn_pos_morph)
            try:
                udpos, _ = cgn2udpos[cgn_tag]
            except KeyError:
                print(token, cgn_tag)
                raise RuntimeError("No UD pos found for {}".format(cgn_tag))
            if udpos not in relevant_udpos:
                continue
            token_ = to_ascii(token)
            if re.match(r"^[0-9a-z'&-]+$", token_, flags=re.IGNORECASE):
                digit_proportion = (
                    sum(1 for char in token_ if char.isdigit())
                    / 
                    len(token_))
                if digit_proportion > 0.333:
                    continue
                elif udpos == "ADJ" and cgn_pos_head == "WW":
                    continue
                elif udpos in ("PUNCT", "ADP", "PRON", "DET"):
                    pass
                elif len(token) > min_token_length:
                    pass
                else:
                    continue
            token, lemma = token.lower(), lemma.lower()

            old_entry = token_lemma_mapping_index.get((token, udpos))
            if old_entry:
                if freq > old_entry[2]:
                    token_lemma_mapping_index[(token, udpos)] = (lemma, cgn_tag, freq)
            else:
                token_lemma_mapping_index[(token, udpos)] = (lemma, cgn_tag, freq)

    token_lemma_pos_pairs = [(tok, lem, pos, cgn, freq)
                             for (tok, pos), (lem, cgn, freq) in token_lemma_mapping_index.items()]

    with open(out_f, "wt") as fp:
        writer = csv.writer(fp, delimiter="\t")
        for el in token_lemma_pos_pairs:
            writer.writerow(el)


if __name__ == '__main__':
    plac.call(main)

