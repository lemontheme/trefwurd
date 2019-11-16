import functools as ft
import itertools as it
import logging
import operator as op
import random
import re
from collections import deque, defaultdict
from difflib import SequenceMatcher
from math import log2
from pprint import pprint
from typing import (
    NewType,
    Tuple,
    Iterable,
    Sequence,
    Pattern as RePattern,
    Optional,
    Match as ReMatch,
    Dict,
    Set,
    Union,
    Iterator,
    Callable,
    List,
    Any,
    NamedTuple)
from toolz import compose as compose_funcs

from tqdm import tqdm

from trefwurd.rules import Rule, ExamplePair, prime_rules
from trefwurd.tree import RuleTree, RuleNode
from trefwurd.utils import binary_entropy, smoothed_probs


logger = logging.getLogger(__name__)


def token_lemma_pairs() -> List[Tuple[str, str]]:
    return [
        ("geknipt", "knippen"),  # ge*t -> *pen
        ("gelapt", "lappen"),  # ge*t -> *pen
        ("geschikt", "schikken"),  # ge*t -> ge*ken
        ("gekaapt", "kapen"),  # ge*apt -> *pen
        ("wandelden", "wandelen"),  # *den -> *en
        ("afgevraagd", "afvragen"),
        ("praat", "praten"),
        ("denken", "denken"),
        ("plink", "barr"),
        ("gezegd", "zeggen"),
        ("bindt", "binden"),
        ("gewipt", "wippen")
    ]


def get_examples() -> List[ExamplePair]:
    return [ExamplePair(tok, lem) for tok, lem in token_lemma_pairs()]


def test_str_tree_builder():
    str_tree = RuleTree()
    str_tree.fit(*zip(*token_lemma_pairs()))
    pprint([n.rule for n in str_tree.root])
    for tok, _ in token_lemma_pairs():
        print(tok, str_tree.predict(tok))
    # print("result:", str_tree.predict("fdsfsdf"))


def test_collect_prime_rules():
    examples = get_examples()
    rules = list(prime_rules(examples))
    pprint(rules)


def test_impurity():
    print(binary_entropy((1, 54)))


def test_get_leaves():
    n = [RuleNode(id_=i) for i in range(4)]
    n[1].attach_to(n[0], branch="left")
    n[2].attach_to(n[0], branch="right")
    n[3].attach_to(n[2], branch="left")
    # print([id(n_) for n_ in n])
    # print([id(n_) for n_ in n[0].leaves])
    print(n[0].display())


def test_detach_from_copy():
    import copy
    n = [RuleNode() for _ in range(4)]
    n[1].attach_to(n[0], branch="left")
    n[2].attach_to(n[1], branch="left")
    tree = n[0]
    tree_copy = copy.deepcopy(tree)


def test_serialize_and_deserialize_tree():
    n = [RuleNode(id_=i) for i in range(4)]
    n[1].attach_to(n[0], branch="left")
    n[2].attach_to(n[0], branch="right")
    n[3].attach_to(n[2], branch="left")
    tree = RuleTree(n[0])
    as_dict = tree.to_dict()
    print(as_dict)
    new_tree = RuleTree.from_dict(as_dict)
    print(new_tree.to_dict())


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # test_str_tree_builder()
    # test_impurity()
    # test_collect_prime_rules()
    # test_detach_from_copy()
    # test_get_leaves()
    test_serialize_and_deserialize_tree()

    # from trefwurd.rules import CountsTable
    # ct = CountsTable(*range(6))
    # print(ct.c_total)

# def child_rule_test():
#     vocab = [ExamplePair(t, l) for t, l in test_cases()]
#     root = make_root_rule()
#     root_stats = evaluate_rule(root, vocab)
#     exhausted = set()
#     for child_rule, child_rule_stats in child_rule_generator(
#             root, root_stats, exhausted, child_gen_f=prime_rule_generator
#     ):
#         print(child_rule, child_rule_stats.summary)
#
#
# def train_tree_test():
#     nodes.jsonl = train_string_transformation_rule_tree(test_cases(), max_depth=10, max_nodes=1000)
#     print("result:")
#     import pprint
#
#     pprint.pprint(nodes.jsonl)
#
#
# def test_str_tree():
#     str_tree = STRtree()
#     token_lemma_pairs = test_cases()
#     str_tree.fit(token_lemma_pairs)
#
#
# def test_str_tree_serialize():
#     str_tree = STRtree()
#     token_lemma_pairs = test_cases()
#     str_tree.fit(token_lemma_pairs)
#     print(str_tree.serialize())
#
#
# def lemmatizer_test_cases():
#     x, y = [], []
#     for token, lemma in test_cases():
#         x.append((token, "VERB"))
#         y.append(lemma)
#     return x, y
#
#
# def test_lemmatizer():
#     lemmatizer = Lemmatizer()
#     x, y = lemmatizer_test_cases()
#     lemmatizer.fit(x, y)
#     print(lemmatizer.lemmatize("gezegd", "VERB"))
#
#
# def test_train_decision_tree():
#     tok_lem_pairs = test_cases()
#     print(tok_lem_pairs)
#     import pprint
#
#     print("RESULT")
#     pprint.pprint(train_decision_tree(tok_lem_pairs))
#
#
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     # print(derive_new_rule(Rule("*d", "*en"), ExamplePair("afgevraagd", "afvragen"), 1))
#     # print(derive_new_rule(Rule("*t", "*en"), ExamplePair("praat", "praten"), 1))
#     # print(lhs_rhs_patterns("***ge***d", "******gen"))
#
#     # x = ExamplePair("regent", "regenen")
#     # print(prime_rule(x))
#
#     # print(is_subsumed_by("*a*t", "*t"))
#
#     # rn1 = RuleNode(Rule("*t", "*en"))
#     # rn2 = RuleNode(Rule("*at", "*ten"))
#     # rn1.attach(child=rn2)
#
#     # child_rule_test()
#
#     # WORKS as expected.
#     # parent_support = SupportStats(correct={"a", "b"})
#     # child_support = SupportStats(correct={"b"})
#     # _transfer_parent_support_to_child_rule(
#     #     parent_support,
#     #     child_support
#     # )
#     # print(parent_support)
#     # print(child_support)
#
#     # s1 = {ExamplePair("regent", "regenen"), ExamplePair("wandelt", "wandelen"), ExamplePair("zingt", "zingen")}
#     # s2 = {ExamplePair("wandelt", "wandelen")}
#     #
#     # stats1 = SupportStats(correct=s1)
#     # stats2 = SupportStats(correct=s2, incorrect={ExamplePair("zingt", "zingen")})
#     # _transfer_parent_support_to_child_rule(stats1, stats2)
#     # print(stats1)
#     # print(stats2)
#     #
#     # train_tree_test()
#     # test_str_tree()
#     # test_lemmatizer()
#     # test_str_tree_serialize()
#     test_train_decision_tree()
