import sys
import csv
import os
import re
import json
import random
import pprint
import logging
from collections import defaultdict, ChainMap, Counter
from copy import deepcopy
import itertools as it
import functools as ft
import gzip

IRREGULAR_VERBS_F = "data/external/nl_irregular_verbs.json"
IRREGULAR_NOUNS_F = "data/external/nl_irregular_plural_nouns.json"
ELEX_TOK_LEM_POS_F = "data/processed/word_lemma_pos_elex.txt"  # tab-delimited cols
# Supplementary word list. Noisier than eLex, so in case of conflicts eLex always wins.
# WIKI_FROGGED_TOK_LEM_POS_F = "../data/interim/word_lemma_pos_frogged_wiki.txt"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)




if __name__ == '__main__':
    read_lexical_data(ELEX_TOK_LEM_POS_F)



