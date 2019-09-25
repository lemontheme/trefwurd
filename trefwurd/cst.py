"""
CST
===

Replication of CST algorithm.

- NOT the same as lemmy
- Different than affixtrain (cpp implementation)

Glossary:

- 'LHS' = left-hand side (of rule)
- 'RHS' = right-hand side (of rule)
- 'srt rule' = string transformation rule

"""

import functools as ft
import itertools as it
import logging
import operator as op
import random
import re
from collections import deque, defaultdict
from difflib import SequenceMatcher
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
)

logger = logging.getLogger(__name__)


WILDCARD_CHAR = "*"

# TODO: differentiate between masking character (multiple *)
#  (or maybe change to '#') and wildcard/placeholder (single *)

# TODO: Rename Rule to STR (short for 'String Transformation Rule')


class ExamplePair:

    __slots__ = "token", "lemma", "token_mask", "lemma_mask"

    def __init__(self, token: str, lemma: str):
        self.token = token
        self.lemma = lemma
        self.token_mask, self.lemma_mask = compute_overlap_masks(self.token, self.lemma)

    @property
    def is_exhausted(self):
        return not any(self.token_mask)

    @property
    def overlap_masked(self):
        return apply_mask(self.token, self.token_mask), apply_mask(self.lemma, self.lemma_mask)

    def __hash__(self):
        return hash((self.token, self.lemma))

    def __eq__(self, other):
        return self.token, self.lemma == self.token, self.lemma

    def __repr__(self):
        return f"<{type(self).__name__} obj @ {id(self)}; token={self.token} lemma={self.lemma}>"


Vocab = NewType("Vocab", Iterable[ExamplePair])


def compute_overlap_masks(seq1, seq2) -> Tuple[bytearray, bytearray]:
    overlap = sequence_overlap(seq1, seq2)
    seq1_mask = mask_from_spans(((s_idx, s_idx + size) for s_idx, _, size in overlap), len(seq1))
    seq2_mask = mask_from_spans(((s_idx, s_idx + size) for _, s_idx, size in overlap), len(seq2))
    return seq1_mask, seq2_mask


def sequence_overlap(seq1: Sequence, seq2: Sequence):
    """Convenience function for computing overlap between two sequences.

    Returns:
         Tuples of the form (start idx in seq1, start idx in seq2, size/length).
    """
    diff = SequenceMatcher(None, seq1, seq2)
    # Drop last matching block, since this is always a dummy entry, with length=1.
    return diff.get_matching_blocks()[:-1]


def mask_from_spans(spans: Iterable[Tuple[int, int]], seq_length: int) -> bytearray:
    mask = bytearray(seq_length)
    for s_i, e_i in spans:
        mask[s_i:e_i] = it.repeat(1, e_i - s_i)
    return mask


def invert_mask(mask: bytearray) -> bytearray:
    return bytearray(1 ^ i for i in mask)  # `^` = XOR


def apply_mask(s: str, mask: bytearray):
    return "".join("*" if m else x for x, m in zip(s, mask))


def collapse_wildcards(s: str) -> str:
    return re.sub(f"{re.escape(WILDCARD_CHAR)}+", WILDCARD_CHAR, s)


class Rule:

    lhs: str  # "ge*t"
    rhs: str  # "*en"
    _match_re: RePattern  # alias re.Pattern
    _sub_regexp: str

    __slots__ = "lhs", "rhs", "_match_re", "_sub_regexp"

    def __init__(self, lhs: str, rhs: str):
        """
        >>> Rule("ge*t", "**en")
        """
        self.lhs, self.rhs = lhs, rhs
        match_regexp, self._sub_regexp = rule_as_regexp(lhs, rhs)
        self._match_re = re.compile(match_regexp)

    def apply(self, s: str) -> Optional[str]:
        m: ReMatch = self.match(s)
        if m:
            return self.transform_match(m)

    def match(self, s: str) -> Optional[ReMatch]:
        return self._match_re.fullmatch(s)

    def transform_match(self, m: ReMatch) -> str:
        return m.expand(self._sub_regexp)

    def __eq__(self, other: "Rule"):
        return self.lhs, self.rhs == other.lhs, other.rhs

    def __lt__(self, other: "Rule"):
        return is_subsumed_by(self.lhs, other.lhs)

    def __gt__(self, other: "Rule"):
        return is_subsumed_by(other.lhs, self.lhs)

    def __repr__(self):
        return f"<{type(self).__name__} obj @ {id(self)}; {self.as_string()}>"

    def as_string(self):
        return f"{self.lhs}-->{self.rhs}"

    def __hash__(self):
        return hash((self.lhs, self.rhs))


def select_rule(rule_tree, s: str) -> Tuple[Rule, ReMatch]:
    pass


def rule_as_regexp(rule_lhs_pattern: str, rule_rhs_pattern: str):  # -> Callable[[str], str]:
    """
    Note:
        Rule pattern uses "*" to mean '__one__ or more characters'.
    """
    lhs_wildcards_n = sum(1 for s in rule_lhs_pattern if s == WILDCARD_CHAR)
    rhs_wildcards_n = sum(1 for s in rule_rhs_pattern if s == WILDCARD_CHAR)
    # print(rule_lhs_pattern, rule_rhs_pattern, lhs_wildcards_n, rhs_wildcards_n)
    assert lhs_wildcards_n == rhs_wildcards_n
    match_pattern = regexify_matching_pattern(rule_lhs_pattern)
    sub_pattern = f"{rule_rhs_pattern}"
    for backref_n in range(1, lhs_wildcards_n + 1):
        sub_pattern, _ = re.subn(
            re.escape(WILDCARD_CHAR), r"\\{}".format(backref_n), sub_pattern, count=1
        )
    return match_pattern, sub_pattern


def regexify_matching_pattern(rule_pattern: str, wildcard_optional=False) -> str:
    """Regexifies pattern against which tokens will be matched (i.e. the left-
    hand side of the rule usually).
    """
    return rule_pattern.replace("*", f"(.{'+*'[wildcard_optional]})")


def captured_groups_as_spans(m: ReMatch) -> Sequence[Tuple[int, int]]:
    return [m.span(g) for g in range(1, len(m.groups()) + 1)]


def lhs_rhs_patterns(masked_token, masked_lemma) -> Tuple[str, str]:
    """pattern = (lhs, rhs). forms basis of Rule.

    Args:
        masked_token: e.g. "***ge***d"
        masked_lemma: e.g. "******gen"
    """
    placeholder_lengths = list(
        sum(1 for _ in v) for k, v in it.groupby(masked_token) if k == WILDCARD_CHAR
    )
    if not (sum(placeholder_lengths) == sum(1 for char in masked_lemma if char == WILDCARD_CHAR)):
        raise ValueError(
            f"Arguments should contain an equal number of wildcards ({WILDCARD_CHAR})."
        )

    token_pattern = re.sub(f"{re.escape(WILDCARD_CHAR)}+", WILDCARD_CHAR, masked_token)

    _lemma = masked_lemma
    _lemma_pattern = []
    for pl in placeholder_lengths:
        lemma_head, lemma_rest = _lemma.split(WILDCARD_CHAR * pl, maxsplit=1)
        _lemma_pattern.append(lemma_head)
        _lemma = lemma_rest
    _lemma_pattern.append(_lemma)
    lemma_pattern = WILDCARD_CHAR.join(_lemma_pattern)

    return token_pattern, lemma_pattern


def is_subsumed_by(rule_pattern1, rule_pattern2):
    """
    Return `True` if rule_pattern2 is a generalization of rule_pattern1,
    or, equivalently, if rule_pattern1 is a more constrained version of
    rule_pattern2.
    """
    if rule_pattern1 == rule_pattern2:
        return False
    if re.match(regexify_matching_pattern(rule_pattern2), rule_pattern1):
        return True
    else:
        return False


# ------------------------------------------------------------------------------


class RuleNode:
    def __init__(
        self,
        rule: Rule,
        parent: Optional["RuleNode"] = None,
        children: Optional[Sequence["RuleNode"]] = None,
        id_=None,
    ):
        self.id_ = id_
        self.rule = rule
        self.parent = parent
        self.children = children or []

    def attach(
        self, parent: Optional["RuleNode"] = None, children: Iterable["RuleNode"] = None
    ) -> Optional["RuleNode"]:
        if parent and children:
            raise ValueError
        if parent:
            if self not in parent.children:
                parent.children.append(self)
                self.parent = parent
                return self
        elif children:
            for child in children:
                if child.parent is None and child not in self.children:
                    child.parent = self
                    self.children.append(child)
                    return self
        else:
            raise ValueError("No attachment target provided.")

    def detach(
        self, parent: Optional["RuleNode"] = None, children: Iterable["RuleNode"] = None
    ) -> Optional["RuleNode"]:
        if parent and children:
            raise ValueError
        if parent:
            try:
                parent.children.remove(self)
                self.parent = None
                return self
            except ValueError:
                return None  # Raised by p.c.remove(x) if x not found.
        elif children:
            for child in children:
                if child.parent is self:
                    try:
                        self.children.remove(child)
                        child.parent = None
                    except ValueError:
                        return
        else:
            raise ValueError("No target for detaching provided.")

    @property
    def ancestors(self) -> Sequence["RuleNode"]:
        ancestors = []
        _ancestor = self.parent
        while _ancestor:
            ancestors.append(_ancestor)
            _ancestor = _ancestor.parent
        return ancestors

    @property
    def depth(self) -> int:
        """Counting from 0, such that root node's depth is equal to 0."""
        return len(self.ancestors)

    # def __eq__(self, other: "RuleNode"):
    #     return self.rule == other.rule

    def __repr__(self):
        return (
            f"<{type(self).__name__} obj @ {id(self)}; id={self.id_}, rule={self.rule.as_string()}, "
            f"parent={self.parent.rule.as_string() if self.parent else None}, children={len(self.children)}>"
        )

    def __hash__(self):
        return id(self) ^ hash(self.rule)


# Rule generation --------------------------------------------------------------

ExhaustedExamples = Set[ExamplePair]


class TokenMaskExhaustedException(Exception):
    """Signal that a token mask required all its masking bits to be set to 0 in
    order for the (token, lemma) combination to give rise to a more specific
    rule. At this point, if the token...
    """

    def __init__(self, rule=None, example=None):
        self.rule = rule
        self.example = example


def prime_rule(example: ExamplePair) -> Rule:
    rule = Rule(*lhs_rhs_patterns(*example.overlap_masked))
    if example.is_exhausted:
        raise TokenMaskExhaustedException(rule, example)
    else:
        return rule


def derive_new_rule(rule: Rule, example: ExamplePair, max_changes=1) -> Rule:
    """Derive new rule from RULE that incorrectly lemmatized TOKEN.

    The (lhs, rhs) patterns of the new rule are minimally different from those
    of the original rule. A number of extension strategies are permitted:

    - Add an extra literal character;
    - Remove a wildcard;
    - Replace a wildcard with one literal character.

    Returns:
        Rule
    """
    # Determine how pattern in rule.lhs matched token.
    token, lemma = example.token, example.lemma
    m = rule.match(token)
    if not m:
        raise ValueError("Provided token did not match left-hand side of rule.")
    # Start and end indices of subsequences that matched wildcards (*).
    token_placeholder_spans = captured_groups_as_spans(m)
    # Reverse engineered: token_mask that could have generated rule.
    token_mask = mask_from_spans(token_placeholder_spans, len(token))
    # Character indexes that are masked by token_mask (have value '1')
    masked_idxs = [i for i in range(len(token)) if token_mask[i] == 1]
    # All possible combinations of masking idxs that will be unmasked.
    derived_rule_candidates = []
    # Token mask hypotheses based on 'flipping' one or n < max_changes idxs in
    # `masked_idxs`.
    for unmasking_hyp in it.chain.from_iterable(
        it.combinations(masked_idxs, n + 1) for n in range(max_changes)
    ):
        new_token_mask = token_mask.copy()
        for idx in unmasking_hyp:
            new_token_mask[idx] = 0
        new_lemma_mask = _infer_corresponding_lemma_mask(token, new_token_mask, lemma)
        if new_lemma_mask:
            new_rule = Rule(
                *lhs_rhs_patterns(
                    apply_mask(token, new_token_mask), apply_mask(lemma, new_lemma_mask)
                )
            )
            derived_rule_candidates.append((new_rule, new_token_mask))
    # If valid rules found by first strategy, select the rule whose lhs differs
    # the least from the lhs of the original rule.
    if derived_rule_candidates:
        # select least different
        sm = SequenceMatcher(None)
        sm.set_seq2(rule.lhs)
        most_sim_dr_i, max_sim = 0, 0
        for dr_i, (dr_rule, dr_token_mask) in enumerate(derived_rule_candidates):
            sm.set_seq1(dr_rule.lhs)
            similarity = sm.ratio()
            if similarity > max_sim:
                max_sim = similarity
                most_sim_dr_i = dr_i
        best_rule, underlying_token_mask = derived_rule_candidates[most_sim_dr_i]
    # Fallback to prime rule lhs merged with token_mask according to lhs of current rule.
    else:
        new_token_mask = underlying_token_mask = bytearray(
            map(op.and_, token_mask, example.token_mask)
        )
        new_lemma_mask = _infer_corresponding_lemma_mask(token, new_token_mask, lemma)
        best_rule = Rule(
            *lhs_rhs_patterns(apply_mask(token, new_token_mask), apply_mask(lemma, new_lemma_mask))
        )
    if not any(underlying_token_mask):
        raise TokenMaskExhaustedException(best_rule, example)
    else:
        return best_rule


def _infer_corresponding_lemma_mask(token, token_mask, lemma) -> Optional[bytearray]:
    token_mask_inverted = invert_mask(token_mask)
    # Re-mask token according to inverted modified character mask.
    # This unmasks the characters that would otherwise be masked; and masks
    # those characters which are normally visible in the rule pattern.
    _new_masked_token = apply_mask(token, token_mask_inverted)
    # Transform _masked_token into usable _pattern.
    _pattern = regexify_matching_pattern(
        collapse_wildcards(_new_masked_token), wildcard_optional=True
    )
    # Match pattern against lemma.
    m = re.fullmatch(_pattern, lemma)
    if m:
        _new_lemma_mask = invert_mask(mask_from_spans(captured_groups_as_spans(m), len(lemma)))
        return _new_lemma_mask


# TODO: Use a dataclass for this instead
class SupportStats:

    __slots__ = "correct", "incorrect", "not_matched"

    correct: Set[ExamplePair]
    incorrect: Set[ExamplePair]
    not_matched: Set[ExamplePair]

    def __init__(self, correct=None, incorrect=None, not_matched=None):
        self.incorrect = incorrect or set()
        self.not_matched = not_matched or set()
        self.correct = correct or set()

    @property
    def matched(self):
        return self.correct | self.incorrect  # set union

    @property
    def n_correct(self):
        return len(self.correct)

    @property
    def n_incorrect(self):
        return len(self.incorrect)

    @property
    def n_not_matched(self):
        return len(self.not_matched)

    @property
    def n_matched(self):
        return len(self.matched)

    @property
    def summary(self):
        return self.n_correct, self.n_incorrect, self.n_not_matched

    def __repr__(self):
        return f"{type(self).__name__}({', '.join('{}={}'.format(k, getattr(self, k)) for k in self.__slots__)})"


def prime_rule_generator(vocab: Vocab, rule: Rule, exhausted: ExhaustedExamples, *args):
    _ = rule
    for x in vocab:
        try:
            yield prime_rule(x)
        except TokenMaskExhaustedException:
            exhausted.add(x)


def derived_rule_generator(
    vocab: Vocab, rule: Rule, exhausted: ExhaustedExamples, max_changes=1, *args
):
    for x in vocab:
        try:
            yield derive_new_rule(rule, x, max_changes=max_changes)
        except TokenMaskExhaustedException:
            exhausted.add(x)


# def _transfer_parent_support_to_child_rule(
#     parent_support: SupportStats, child_support: SupportStats
# ) -> None:
#     parent_support.correct.difference_update(child_support.correct)
#     parent_support.incorrect.difference_update(child_support.incorrect)
#
#
# def _update_child_support_from_parent(
#     child_support: SupportStats, parent_support: SupportStats
# ) -> None:
#     child_support.correct.intersection_update(parent_support.correct)
#     child_support.incorrect.intersection_update(parent_support.incorrect)


def evaluate_rule(rule: Rule, vocab: Vocab) -> SupportStats:
    ss = SupportStats()
    correct, incorrect, not_matched = ss.correct, ss.incorrect, ss.not_matched
    for x in vocab:
        yhat = rule.apply(x.token)
        if yhat is None:
            not_matched.add(x)
        elif yhat == x.lemma:
            correct.add(x)
        else:
            incorrect.add(x)
    return ss


# Function to be argmax'd
def rule_candidate_contribution_measure(
    child_support: SupportStats, parent_support: SupportStats
) -> Tuple[int, ...]:
    """
    RETURNS tuple of numeric values expressing positive contribution of rule. Higher
    means better.
    """
    # Matched at least one example.
    c0 = int(bool(child_support.n_matched))
    # N_wr + N_rr - N_rw (r := right; w := wrong)
    c1 = (
        len(parent_support.incorrect | child_support.correct)
        + len(parent_support.correct | child_support.correct)
        - len(parent_support.correct | child_support.incorrect)
    )
    # Select for lowest N_rr. Flipping sign so as to be able to minimize by
    # argmaxing.
    c2 = -(len(parent_support.correct | child_support.correct))
    # N_rn - N_ww
    c3 = len(parent_support.correct | child_support.not_matched) - len(
        parent_support.incorrect | child_support.incorrect
    )
    # Simple tie breaker.
    c4 = int(100 * random.random())
    return c0, c1, c2, c3, c4


def child_rule_generator(
    rule: Rule,
    support: SupportStats,
    exhausted: ExhaustedExamples = None,
    child_gen_f: Callable[
        [Vocab, Rule, ExhaustedExamples], Iterator[Rule]
    ] = derived_rule_generator,
) -> Iterator[Tuple[Rule, SupportStats]]:
    parent_support = support
    parent_rule = rule
    vocab = support.matched
    exhausted = exhausted if exhausted is not None else set()
    child_rule_candidates: Dict[Rule, SupportStats] = {}
    child_rule_candidates.update(
        (r, evaluate_rule(r, vocab))
        for r in child_gen_f(vocab, parent_rule, exhausted)
        if r not in child_rule_candidates
    )
    while child_rule_candidates and parent_support.n_matched > 0:
        # print([x.summary for x in child_rule_candidates.values()])
        best_child_rule, best_child_support = max(
            child_rule_candidates.items(),
            key=lambda tup: rule_candidate_contribution_measure(
                child_support=tup[1], parent_support=parent_support
            ),
        )
        # print(best_child_rule, f"({[x.token for x in best_child_support.matched]})", best_child_support.n_matched)
        if best_child_support.n_matched == 0:
            return child_rule_candidates

        # Remove selected rule from future consideration.
        child_rule_candidates.pop(best_child_rule)

        # Remove rules from consideration with the same left-hand side (pattern
        # that matches incoming tokens) as that of the selected child rule.
        _best_child_rule_lhs_pattern = best_child_rule.lhs
        candidates_w_same_lhs = [
            r for r in child_rule_candidates.keys() if r.lhs == _best_child_rule_lhs_pattern
        ]
        for r in candidates_w_same_lhs:
            del child_rule_candidates[r]
        # Transfer parent support to child_rule
        # (Previously performed by calling
        #  _transfer_parent_support_to_child_rule(parent_support, best_child_support)
        # )
        matched_by_child = best_child_support.matched
        parent_support.correct.difference_update(matched_by_child)
        parent_support.incorrect.difference_update(matched_by_child)
        # Update children support from parent and sibling
        # (Previously performed by calling
        # _update_child_support_from_parent(other_child_support, parent_support)
        # )
        for other_child_support in child_rule_candidates.values():
            other_child_support.correct.intersection_update(parent_support.correct)
            other_child_support.incorrect.intersection_update(parent_support.incorrect)
            other_child_support.not_matched.intersection_update(best_child_support.not_matched)
        yield best_child_rule, best_child_support


def make_root_rule():
    return Rule(WILDCARD_CHAR, WILDCARD_CHAR)


ChildRuleGenerator = Iterator[Tuple[Rule, SupportStats]]


def train_string_transformation_rule_tree(
    token_lemma_pairs: Iterable[Tuple[str, str]], max_depth=1000, max_nodes=-1
) -> Tuple[Sequence[RuleNode], Dict[str, str], Tuple[int, int, int]]:

    # TODO: implement proper stopping criterion.

    max_nodes = max_nodes if max_nodes > 0 else 9999

    vocab: Vocab = [ExamplePair(*tok_lem) for tok_lem in token_lemma_pairs]
    exhausted: ExhaustedExamples = set()

    root_node = RuleNode(make_root_rule(), id_=0)
    root_node_support = evaluate_rule(root_node.rule, vocab)
    root_node_child_rules_iter = child_rule_generator(
        root_node.rule, root_node_support, exhausted, prime_rule_generator
    )

    expandable: List[Tuple[RuleNode, SupportStats, ChildRuleGenerator]] = []
    non_expandable: List[Tuple[RuleNode, SupportStats, ChildRuleGenerator]] = []
    expandable.append((root_node, root_node_support, root_node_child_rules_iter))

    def total_nodes():
        return len(expandable) + len(non_expandable)

    def global_performance() -> Tuple[int, int, int]:
        """
        RETURNS: (correct, incorrect, non_matched)
        """
        make_nodes_support_iter = lambda: map(
            op.itemgetter(1), it.chain(expandable, non_expandable)
        )
        correct = ft.reduce(set.union, (support.correct for support in make_nodes_support_iter()))
        incorrect = ft.reduce(
            set.union, (support.incorrect for support in make_nodes_support_iter())
        )
        if correct & incorrect:
            raise RuntimeError(
                "There should be no overlap between the sets of "
                "instances of correct and incorrect lemmatization."
            )
        n_correct, n_incorrect = len(correct), len(incorrect)
        n_not_matched = sum(1 for x in vocab if x not in (correct | incorrect))
        return n_correct, n_incorrect, n_not_matched

    # Record of global performance at each training step.
    history = [global_performance()]

    def cease_training() -> bool:
        if len(history) > 1 and history[-2][1] < history[-1][1]:
            logger.debug("Terminate training: Previous step made as many/fewer errors.")
            return True
        if total_nodes() >= max_nodes:
            logger.debug("Terminate training: Reached maximum number of nodes in tree.")
            return True

    new_nodes_counter = total_nodes()

    while not cease_training():
        if not expandable:
            break

        expandable.sort(key=lambda x: (x[1].n_incorrect, random.random()))

        while expandable:

            node, node_support, node_child_iter = n = expandable.pop()
            if node.depth == max_depth:
                logger.debug(f"Not extending: Hit max. node depth d={max_depth}")
            elif node_support.n_incorrect == 0:
                logger.debug(f"Not extending: Node {node} already has zero error.")
            else:
                try:
                    child_rule, child_rule_support = next(node_child_iter)
                except StopIteration:
                    logger.debug(f"Not extending: Cannot derive any more children from {node.rule}")
                else:
                    expandable.append(n)
                    break  # Signals that the node was expandable. (Skips ELSE block.)
            non_expandable.append(n)
        else:  # Block executed only if all expandable nodes were exhausted.
            continue

        # ----------------------------------------------------------------------
        # Everything beyond this point is reachable only if next(node_child_iter)
        # yielded a child_rule.
        child_rule_node = RuleNode(child_rule, id_=new_nodes_counter)
        child_rule_node.attach(parent=node)
        child_rule_child_rules_iter = child_rule_generator(
            child_rule, child_rule_support, exhausted, child_gen_f=derived_rule_generator
        )
        expandable.append((child_rule_node, child_rule_support, child_rule_child_rules_iter))

        new_nodes_counter += 1
        history.append(global_performance())

    nodes_only = sorted(
        map(op.itemgetter(0), it.chain(expandable, non_expandable)), key=lambda t: t.id_
    )

    exhausted_as_lookup = {x.token: x.lemma for x in exhausted}

    # First item in `nodes_only` is the root node.
    return nodes_only, exhausted_as_lookup, history[-1]


class STRtree:
    def __init__(self, **cfg):
        self.nodes = None
        self.root = None
        self.cfg = cfg
        self._exhausted_lookup = None

    @staticmethod
    def _find_and_apply_matching_rule(
        tree: RuleNode, s: str
    ) -> Tuple[Optional[str], Optional[RuleNode]]:
        """Find deepest, leftmost rule node that matches `s` and return the
        result of rule application, accompanied by the node hosting the rule.

        RETURN: (result of applying rule, rule node)
        """
        match, rule_node = None, None
        queue = deque([tree])
        while queue:
            node = queue.popleft()
            m = node.rule.match(s)
            if m:
                match = m
                rule_node = node
                queue.clear()
                queue.extend(rule_node.children)
        result = rule_node.rule.transform_match(match) if match else None
        return result, rule_node

    def fit(self, x: Iterable[Tuple[str, str]]) -> "STRtree":
        nodes, exhausted, _ = train_string_transformation_rule_tree(x, **self.cfg)
        self.nodes = nodes
        self.root = nodes[0]
        self._exhausted_lookup = exhausted

    def predict(self, s: str) -> str:
        try:
            result, _ = self._find_and_apply_matching_rule(self.root, s)
            return result
        except AttributeError:
            raise RuntimeError("Method called that required STRtree to be fit() first.")

    def lookup(self, s):
        try:
            return self._exhausted_lookup.get(s, None)
        except AttributeError:
            raise RuntimeError("Method called that required STRtree to be fit() first.")

    def serialize(self):
        return self._serialize_nodes()

    def _serialize_nodes(self):
        """RETURNS: (serialized_nodes, serialized_lookup"""
        node_lines = []
        node: RuleNode
        for node in self.nodes:
            node_lines.append(
                "\t".join(
                    (str(node.id_),  # node ID
                     node.rule.as_string(),  # node rule (e.g., "*d->*en")
                     str(node.parent.id_ if node.parent else "_"),  # parent node ID or empty string
                     (",".join(str(n.id_) for n in node.children) or "_"))
                )
            )
        return "\n".join(node_lines)

    @staticmethod
    def deserialize(
        self, nodes_data: str, lookup_data: Optional[str] = None
    ) -> Tuple[Sequence[RuleNode], Optional[Dict[str, str]]]:
        """RETURNS: (nodes, lookup)"""
        pass

    @classmethod
    def load(cls, fp) -> "STRtree":
        pass

    def save(self, f) -> None:
        pass


def serialize_strtree_nodes(nodes: Sequence[RuleNode]) -> str:
    pass


def serialize_strtree_lookup(lookup: Dict[str, str]) -> str:
    pass


def parse_serialized_strtree_nodes(data: str) -> Sequence[RuleNode]:
    pass


def parse_serialized_strtree_lookup() ->:
    pass


class Lemmatizer:
    def __init__(self, pos=True, morph=False, cache=False, **cfg):
        self.trees: Dict[Any, STRtree] = {}
        self.cfg = cfg
        self._lemmatize = None
        self._only_tree: Optional[STRtree] = None
        self._use_pos = pos
        self._use_morph = morph
        if morph:
            raise ValueError("Morphological features currently not supported.")

    def fit(
        self,
        x: Sequence[Union[str, Tuple[str, str], Tuple[str, str, Tuple[str, ...]]]],
        y: Sequence[str],
    ) -> "Lemmatizer":
        """
        Args:
            x: (tokens, *features [^1])
            y: lemmas
        Returns: Lemmatizer
        [1]: usually just pos
        """
        if not len(x) == len(y):
            raise ValueError("`x` and `y` args should have the same length.")
        if self._use_pos:
            if self._use_morph:
                self._fit_w_morph()
            else:
                self._fit_w_pos(x, y)
        else:
            self._fit_no_pos(x, y)
        return self

    def _fit_no_pos(self, x: Sequence[str], y: Sequence[str]):
        str_tree = STRtree(**self.cfg)
        str_tree.fit(zip(x, y))
        self.trees[None] = str_tree
        self._only_tree = str_tree
        self.lemmatize = self._lemmatize_no_pos

    def _fit_w_pos(self, x: Sequence[Tuple[str, str]], y: Sequence[str]):
        pos2data = defaultdict(list)
        for (token, pos), lemma in zip(x, y):
            pos2data[pos].append((token, lemma))
        for pos, data in pos2data.items():
            str_tree = STRtree(**self.cfg)
            str_tree.fit(data)
            self.trees[pos] = str_tree
        self.lemmatize = self._lemmatize_w_pos

    def _fit_w_morph(self):
        raise NotImplementedError

    def lemmatize(self, s, *args):
        raise NotImplementedError("Concrete implementation set by calling fit() first.")

    def _lemmatize_no_pos(self, s):
        return self._only_tree.predict(s)

    def _lemmatize_w_pos(self, s, pos):
        try:
            return self.trees[pos].predict(s)
        except KeyError:
            return s

    def _lemmatize_w_morph(self, pos, morph=None):
        raise NotImplementedError

    def set_logger(self, fp):
        pass


def test_cases():
    return [
        ("geknipt", "knippen"),  # ge*t -> *pen
        ("gelapt", "lappen"),  # ge*t -> *pen
        ("geschikt", "schikken"),  # ge*t -> ge*ken
        ("gekaapt", "kapen"),  # ge*apt -> *pen
        ("wandelden", "wandelen"),  # *den -> *en
        ("afgevraagd", "afvragen"),
        ("praat", "praten"),
        ("plink", "barr"),
        ("gezegd", "zeggen"),
        ("bindt", "binden"),
    ]


def child_rule_test():
    vocab = [ExamplePair(t, l) for t, l in test_cases()]
    root = make_root_rule()
    root_stats = evaluate_rule(root, vocab)
    exhausted = set()
    for child_rule, child_rule_stats in child_rule_generator(
        root, root_stats, exhausted, child_gen_f=prime_rule_generator
    ):
        print(child_rule, child_rule_stats.summary)


def train_tree_test():
    nodes = train_string_transformation_rule_tree(test_cases(), max_depth=10, max_nodes=1000)
    print("result:")
    import pprint

    pprint.pprint(nodes)


def test_str_tree():
    str_tree = STRtree()
    token_lemma_pairs = test_cases()
    str_tree.fit(token_lemma_pairs)


def test_str_tree_serialize():
    str_tree = STRtree()
    token_lemma_pairs = test_cases()
    str_tree.fit(token_lemma_pairs)
    print(str_tree.serialize())


def lemmatizer_test_cases():
    x, y = [], []
    for token, lemma in test_cases():
        x.append((token, "VERB"))
        y.append(lemma)
    return x, y


def test_lemmatizer():
    lemmatizer = Lemmatizer()
    x, y = lemmatizer_test_cases()
    lemmatizer.fit(x, y)
    print(lemmatizer.lemmatize("gezegd", "VERB"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # print(derive_new_rule(Rule("*d", "*en"), ExamplePair("afgevraagd", "afvragen"), 1))
    # print(derive_new_rule(Rule("*t", "*en"), ExamplePair("praat", "praten"), 1))
    # print(lhs_rhs_patterns("***ge***d", "******gen"))

    # x = ExamplePair("regent", "regenen")
    # print(prime_rule(x))

    # print(is_subsumed_by("*a*t", "*t"))

    # rn1 = RuleNode(Rule("*t", "*en"))
    # rn2 = RuleNode(Rule("*at", "*ten"))
    # rn1.attach(child=rn2)

    # child_rule_test()

    # WORKS as expected.
    # parent_support = SupportStats(correct={"a", "b"})
    # child_support = SupportStats(correct={"b"})
    # _transfer_parent_support_to_child_rule(
    #     parent_support,
    #     child_support
    # )
    # print(parent_support)
    # print(child_support)

    # s1 = {ExamplePair("regent", "regenen"), ExamplePair("wandelt", "wandelen"), ExamplePair("zingt", "zingen")}
    # s2 = {ExamplePair("wandelt", "wandelen")}
    #
    # stats1 = SupportStats(correct=s1)
    # stats2 = SupportStats(correct=s2, incorrect={ExamplePair("zingt", "zingen")})
    # _transfer_parent_support_to_child_rule(stats1, stats2)
    # print(stats1)
    # print(stats2)
    #
    # train_tree_test()
    # test_str_tree()
    # test_lemmatizer()
    test_str_tree_serialize()
