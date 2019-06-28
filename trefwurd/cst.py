"""
CST
===

Replication of CST algorithm.

- NOT the same as lemmy
- slightly different than affixtrain (cpp implementation)

Glossary:

- 'LHS' = left-hand side (of rule)
- 'RHS' = right-hand side (of rule)

Rule induction
--------------

X = list of (token, pos), y = list of corresponding lemma forms

vocab := List[InflectionPair]
prime_rule_stats := Mapping[rule_lhs, Mapping[rule_rhs, count]]

proc build_tree := takes



Rule application
----------------

"""
import re
import functools as ft
import itertools as it
import logging
from collections import defaultdict
from difflib import SequenceMatcher
from typing import (Tuple, Callable, Iterable, Sequence, List, Pattern as RePattern,
                    ByteString, NamedTuple, Optional, Match as ReMatch, Union)

from trefwurd.dfs import dfs_graph_best_path

logger = logging.getLogger(__name__)


class MaskedWord(NamedTuple):

    string: str
    mask: bytearray

    def update_from_spans(self, spans: Iterable[Tuple[int, int]]):
        for s, e in spans:
            self.mask[s:e] = b"1" * (e - s)

    def flip_mask_bit(self, *idxs: int):
        for idx in idxs:
            self.mask[idx] = 1

    @property
    def masked(self):
        return apply_mask(self.string, self.mask)

    @classmethod
    def make(cls, string, mask_f=lambda s: bytearray(len(s))):
        return MaskedWord(string, mask_f(string))


def apply_mask(seq: Sequence, mask: Sequence):
    return "".join("*" if m else x for x, m in zip(seq, mask))


class ExamplePair:

    token: MaskedWord  # inflected form
    lemma: MaskedWord  # base form
    pos: str

    def __init__(self, token: str, lemma: str, pos: str = None):
        token_mask, lemma_mask = compute_overlap_masks(token, lemma)
        self.token = MaskedWord(token, token_mask)
        self.lemma = MaskedWord(lemma, lemma_mask)
        self.pos = pos

    @property
    def is_exhausted(self):
        """lhs word mask exhausted"""
        return not any(self.token.mask)

    @property
    def masked(self):
        return self.token.masked, self.lemma.masked

    @property
    def as_rule_template(self):
        return RuleTemplate(*map(collapse_wildcards, self.masked))

    def update_mask(self):
        # TODO: implement.
        pass


def compute_overlap_masks(seq1, seq2) -> Tuple[bytearray, bytearray]:
    mask_1 = bytearray(len(seq1))
    mask_2 = bytearray(len(seq2))
    diff = SequenceMatcher(None, seq1, seq2)
    for a_i, b_i, size in diff.get_matching_blocks():
        if size == 0:
            continue
        for x_i, mask in ((a_i, mask_1), (b_i, mask_2)):
            for i in range(x_i, x_i + size):
                mask[i] = 1
    return mask_1, mask_2


def collapse_wildcards(string, wildcard="*") -> str:
    if wildcard != "*":
        raise ValueError("Currently, '*' is only supported wildcard character.")
    return re.sub(r"\*+", "*", string)


class RuleTemplate(NamedTuple):
    """
    Why this class: Implements value comparison based on subsumption.

    *ge*a*d → ***en

    "The  asterisks  are  wildcards  and  placeholders.  The  pattern  on  the
    left  hand  side  contains  three  wildcards,  each  one  corresponding
    to  one  place-holder in the replacement string on the right hand side,
    in  the  same  order.  The  characters  matched  by  a  wildcard  are
    inserted  in  the  place  kept  free  by  the  corresponding
    placeholder  in  the  replace-ment expression." [^1]

    [1]: https://www.aclweb.org/anthology/P09-1017 (3.1)
    """

    lhs: str
    rhs: str

    def __lt__(self, other):
        return subsumes(self.lhs, other.lhs)

    def __gt__(self, other):
        return subsumes(other.lhs, self.lhs)


class Rule(NamedTuple):
    """
    >>> Rule("ge*t", "**en")
    """
    match_re: RePattern  # (re.Pattern)
    sub_pattern: str
    lhs: str
    rhs: str
    id: Optional[int]

    @classmethod
    def from_template(cls, rule: RuleTemplate, id_=None):
        lhs, rhs = rule
        _match_regexp, repl_regexp = rule_as_regexp(lhs, rhs)
        return Rule(re.compile(_match_regexp), repl_regexp, lhs, rhs, id_ or id(rule))

    def apply(self, string) -> Optional["ApplicationResult"]:
        m = self.match_re.fullmatch(string)
        if not m:
            return None
        lemma = m.expand(self.sub_pattern)
        return ApplicationResult(lemma, m, self)


class ApplicationResult:

    __slots__ = ("result", "match", "rule")

    def __init__(self, lemma, m, rule):
        self.result = lemma
        self.match = m
        self.rule = rule

    def inspect(self, source, target=None):
        pass

    # def __repr__(self):
        # return f"{type(self).__name__}({self.result})"


# Shape: ((input*, output**, target***), rule)
# * = token
# ** = predicted lemma
# *** = target lemma or None


class RuleTree:

    def __init__(self, root_rule=RuleTemplate("*", "*")):
        self._root = Rule(*root_rule)
        self.graph = {self._root: []}

    def add_rule(self, rule: Rule, parent: Rule = None):
        self.graph[rule] = []
        if parent:
            siblings = self.graph[parent]
            if rule not in siblings:
                siblings.append(rule)

    def del_rule(self, rule):
        del self.graph[rule]
        for other_rule, other_rule_children in self.graph.items():
            if rule in other_rule_children:
                other_rule_children.remove(other_rule)

    def find_and_apply(self, string: str) -> ApplicationResult:
        best_path = dfs_graph_best_path(self.graph, self._root, lambda rule_: rule_(string))
        if best_path:
            _, application_result = best_path
            return application_result

    def find_by_match(self):
        pass


def lemmatize(rule_tree: RuleTree, string: str) -> Tuple[str, ReMatch, Rule]:
    pass


# there's something in the water


def mask_spans_from_errors(example_pair):
    pass


def derive_new_rule():
    pass


# def reanalyze_wildcard_spans(m: ReMatch) -> Sequence[Tuple[int, int]]:
#     return [m.span i in enumerate()


def find_error_idx(wrong_string, right_string, from_right=True) -> int:
    pass


def rule_as_regexp(rule_lhs_pattern: str, rule_rhs_pattern: str, wildcard="*"):  # -> Callable[[str], str]:
    """
    Note:
        Rule pattern uses "*" to mean '__one__ or more characters'.
    """
    lhs_wildcards_n = sum(1 for s in rule_lhs_pattern if s == wildcard)
    rhs_wildcards_n = sum(1 for s in rule_rhs_pattern if s == wildcard)
    assert lhs_wildcards_n == rhs_wildcards_n
    match_pattern = regexify_pattern_to_match(rule_lhs_pattern)
    sub_pattern = f"{rule_rhs_pattern}"
    for backref_n in range(1, lhs_wildcards_n + 1):
        sub_pattern, _ = re.subn(r"\*", r"\\{}".format(backref_n), sub_pattern, count=1)
    return match_pattern, sub_pattern


def regexify_pattern_to_match(lhs_pattern) -> str:
    return f"{lhs_pattern.replace('*', '(.+)')}"


def apply_rule(match_pattern, sub_pattern, string):
    m = re.match(match_pattern, string)
    if not m:
        return None
    pred = m.expand(sub_pattern)
    spans = (m.span(g) for g in range(1, len(m.groups()) + 1))
    token_mask = mask_from_spans(spans, len(string))
    return pred, token_mask


def capturing_group_spans(m: ReMatch):
    return (m.span(g) for g in range(1, len(m.groups()) + 1))


def mask_from_spans(spans: Iterable[Tuple[int, int]], seq_len: int) -> ByteString:
    mask = bytearray(seq_len)
    for s, e in spans:
        mask[s:e] = b"1" * (e - s)
    return mask


def subsumes(rule_pattern1, rule_pattern2):
    if rule_pattern1 == rule_pattern2:
        return False
    if re.match(regexify_pattern_to_match(rule_pattern2), rule_pattern1):
        return True
    else:
        return False


# ignoring pos for now
def fit(tokens: Iterable[str], lemmas: Iterable[str]):
    """
    training
    """
    vocab = [ExamplePair(token, lemma) for token, lemma in zip(tokens, lemmas)]
    prime_rule_templates = prime_rules(vocab)  # type: List[RuleTemplate]
    rule_tree = RuleTree()
    rule_node_stats = defaultdict(lambda: defaultdict(list))


def collect_rule_stats():
    pass


def select_rule(candidate_rules, rule_stats):
    pass


def prime_rules(vocab: Sequence[ExamplePair]):
    pr = defaultdict(lambda: defaultdict(int))
    for pair in vocab:
        rule = pair.as_rule_template
        pr[rule.lhs][rule.rhs] += 1
    return [RuleTemplate(lhs, max(rhs_counts.items(lambda x: x[1]))[0])
            for lhs, rhs_counts in sorted(pr.items(),
                                          key=lambda x: sum(x[1].values()), reverse=True)]


# ##########################################################


def test():
    eg1 = "geknipt", "knippen"  # ge*t -> *pen
    eg2 = "gelapt", "lappen"  # ge*t -> *pen
    eg3 = "geschikt", "schikken"  # ge*t -> ge*ken
    eg4 = "gekaapt", "kapen"  # ge*apt -> *pen
    eg5 = "wandelden", "wandelen"  # *den -> *en
    # mw = WildcardWord("geschikt", [])
    pass


def find_error_exp():
    r2 = RuleTemplate("*ge*d", "**en")
    rule = Rule.from_template(r2)
    print(rule)
    token = "uitgelegd"
    lemma = "uitleggen"

    result = rule.apply(token)
    predicted_lemma = result.result

    # goal:
    #   align token and predicted lemma
    # e.g.
    #   (uit)ge(leg)d   # token
    #   (uit)__(leg)en  # incorrectly predicted lemma
    # approach:
    #  combine information about spans of matched groups with right-hand side
    #  of rule so as to reconstruct process that yielded lemma prediction.
    matched_spans = capturing_group_spans(result.match)
    matching_blocks = []
    pos_in_str = 0
    for char in rule.rhs:
        if char == "*":
            try:
                s1, e1 = next(matched_spans)
                s2, e2 = (pos_in_str, pos_in_str + e1 - s1)
                matching_blocks.append((s1, e1, s2, e2))
                pos_in_str += e1 - s1
            except StopIteration:
                break
        else:
            pos_in_str += 1
    print(matching_blocks)

    # goal:
    #   find difference, either from left or from right.
    # e.g. (mode=from right)
    # ...uitlegen  # incorrectly predicted lemma
    # ..uitleggen  # target lemma
    #        x+++  # ("+" := agreement; "x" := disagreement)

    j = 0
    for l, r in zip(predicted_lemma[::-1], lemma[::-1]):
        j += 1
        if l != r:
            break
    print(j, len(predicted_lemma) - j, len(lemma) - j)

    left_spans = list(capturing_group_spans(result.match))
    print(apply_mask(token, mask_from_spans(left_spans, len(token))))

    analysis_right = re.match(regexify_pattern_to_match(result.rule.rhs), predicted_lemma)
    print(apply_mask(predicted_lemma, mask_from_spans(capturing_group_spans(analysis_right), len(predicted_lemma))))

    print(predicted_lemma, lemma)
    diff = SequenceMatcher(None, predicted_lemma, lemma)
    for tag, s1, e1, s2, e2 in diff.get_opcodes():
        print(tag, s1, e1, s2, e2)
        # if tag == "insert":
        #     print(tag, predicted[s1:e1], "->", lemma[s2:e2])


if __name__ == '__main__':
    find_error_exp()
