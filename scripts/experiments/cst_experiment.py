import itertools as it
import random
from collections import Counter, defaultdict

from tqdm import tqdm

from trefwurd.cst import subsumes, InflectionPair
from trefwurd.utils import read_lexical_data, character_ngrams


lexical_data_f = "data/processed/word_lemma_pos_elex.txt"
data = read_lexical_data(lexical_data_f)
word_pairs = data.get("verb").items()
word_pairs = random.sample(word_pairs, len(word_pairs))

prime_rules = defaultdict(lambda: defaultdict(int))
for wp in tqdm(word_pairs):
    ip = InflectionPair(*wp)
    lhs, rhs = ip.as_rule
    prime_rules[lhs][rhs] += 1

most_common_lhs = sorted(prime_rules.items(), key=lambda x: sum(x[1].values()), reverse=True)


def candidate_lhs(sorted_lhss, n=10):
    result = []
    for lhs in sorted_lhss:
        for idx, prev_lhs in enumerate(result):
            if subsumes(lhs, prev_lhs):
                break
        else:
            if len(result) < n:
                result.append(lhs)
                print(result)
    return result


top = [lhs for lhs, _ in most_common_lhs[:100] if lhs != "*"]

candidates = candidate_lhs(top, 10)

for c in candidates:
    print(c, prime_rules[c], sep="=>")


# lhs_subsumptions = defaultdict(list)
# for lhs in prime_rules:
#     for other_lhs in prime_rules:
#         if lhs == other_lhs:
#             continue
#         if subsumes(lhs, other_lhs):
#             lhs_subsumptions[lhs].append(other_lhs)
# print(sorted(lhs_subsumptions.items(), key=lambda x: len(x[1]), reverse=False))
# print([lhs for lhs, subsumptions in lhs_subsumptions.items() if subsumptions == ["*"]])






