import itertools as it
import logging
import operator as op
import re
from difflib import SequenceMatcher
from typing import (
    NewType,
    Tuple,
    Iterable,
    Sequence,
    Pattern as RePattern,
    Optional,
    Match as ReMatch,
    Set,
    NamedTuple,
    List,
    Dict,
    Callable,
)

logger = logging.getLogger(__name__)


WILDCARD_CHAR = "*"


class UncollapseableWildcardsException(Exception):
    pass


class TokenMaskExhaustedException(Exception):
    """
    Signal that a token mask required all its masking bits to be set to 0 in
    order for the (token, lemma) combination to give rise to a more specific
    rule. At this point, if the token...
    """


class ExamplePair:

    __slots__ = "token", "lemma", "token_mask", "lemma_mask"

    def __init__(self, token: str, lemma: str):
        self.token = token
        self.lemma = lemma
        self.token_mask, self.lemma_mask = _compute_overlap_masks(self.token, self.lemma)

    @property
    def is_exhausted(self):
        return not any(self.token_mask)

    @property
    def overlap_masked(self):
        return _apply_mask(self.token, self.token_mask), _apply_mask(self.lemma, self.lemma_mask)

    def __hash__(self):
        return hash((self.token, self.lemma))

    def __eq__(self, other):
        return self.token, self.lemma == self.token, self.lemma

    def __repr__(self):
        return f"<{type(self).__name__} obj @ {id(self)}; token={self.token} lemma={self.lemma}>"


Vocab = NewType("Vocab", Iterable[ExamplePair])


class Rule:

    lhs: str  # "ge*t"
    rhs: str  # "*en"
    _match_re: RePattern  # alias re.Pattern
    _sub_regexp: str

    __slots__ = "lhs", "rhs", "_match_re", "_sub_regexp", "is_prime"

    def __init__(self, lhs: str, rhs: str, is_prime=False):
        """
        >>> Rule("ge*t", "**en")
        """
        self.lhs, self.rhs = lhs, rhs
        match_regexp, self._sub_regexp = _rule_as_regexp(lhs, rhs)
        self._match_re = re.compile(match_regexp)
        self.is_prime = is_prime

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
        return _is_subsumed_by(self.lhs, other.lhs)

    def __gt__(self, other: "Rule"):
        return _is_subsumed_by(other.lhs, self.lhs)

    def __repr__(self):
        return f"<{type(self).__name__} obj @ {id(self)}; {self.to_string()}>"

    def to_string(self):
        return f"{self.lhs}~>{self.rhs}"

    @classmethod
    def from_string(cls, rule_as_string) -> "Rule":
        lhs, rhs = rule_as_string.split("~>")
        return cls(lhs, rhs)

    def __hash__(self):
        return hash((self.lhs, self.rhs))


class CountsTable(NamedTuple):

    cc: int  # correct, correct
    ci: int  # correct, incorrect
    ic: int  # incorrect, correct
    ii: int  # incorrect, incorrect
    nmc: int  # not matched, correct
    nmi: int  # not matched, incorrect

    @property
    def c_total(self):  # all corrects
        return self.cc + self.ci

    @property
    def i_total(self):  # all incorrects
        return self.ii + self.ic

    @property
    def nm_total(self):  # all not-matcheds
        return self.nmc + self.nmi

    @property
    def m_total(self):
        return self.c_total + self.i_total

    @property
    def total(self):
        return self.m_total + self.nm_total

    @classmethod
    def make(
        cls,
        correct: Set[ExamplePair],
        incorrect: Set[ExamplePair],
        not_matched: Set[ExamplePair],
        parent_correct: Set[ExamplePair],
        parent_incorrect: Set[ExamplePair],
    ) -> "CountsTable":
        return CountsTable(
            cc=len(correct & parent_correct),
            ci=len(correct & parent_incorrect),
            ic=len(incorrect & parent_correct),
            ii=len(incorrect & parent_incorrect),
            nmc=len(not_matched & parent_correct),
            nmi=len(not_matched & parent_incorrect),
        )


class RuleStatsExamples(NamedTuple):
    correct: Set[ExamplePair]
    incorrect: Set[ExamplePair]
    not_matched: Set[ExamplePair]


class ChildRuleStats:

    __slots__ = "examples", "_parent_examples", "_counts"

    examples: RuleStatsExamples
    _parent_examples: Optional[Tuple[Set[ExamplePair], Set[ExamplePair]]]
    _counts: Optional[CountsTable]

    def __init__(
        self,
        correct,
        incorrect,
        not_matched,
        parent_examples: Tuple[Set[ExamplePair], Set[ExamplePair]] = None,
    ):
        self.examples = RuleStatsExamples(correct, incorrect, not_matched)
        self._parent_examples = parent_examples
        self._counts = None

    def update(
        self,
        matched_by_rule: Set[ExamplePair],
        parent_examples: Tuple[Set[ExamplePair], Set[ExamplePair]],
    ):
        for example_set in self.examples:
            example_set.intersection_update(matched_by_rule)
        self._parent_examples = parent_examples
        self._counts = None

    @property
    def counts(self) -> CountsTable:
        if not self._counts:
            self._counts = CountsTable.make(*self.examples, *self._parent_examples)
        return self._counts

    def get_parent_examples(self):
        return self._parent_examples


def make_rule_candidates(
    correct: Set[ExamplePair],
    incorrect: Set[ExamplePair],
    rule: Rule = None,
    exhausted=None,
    **cfg,
):
    exhausted = exhausted if exhausted else {}
    _ = cfg

    if rule:
        gen_f = derived_rules
    else:
        gen_f = prime_rules

    vocab = correct | incorrect

    parent_examples = (correct, incorrect)

    return [
        (
            rule,
            ChildRuleStats(
                *evaluate_rule(rule, vocab), parent_examples
            ),
        )
        for rule in gen_f(vocab, rule, exhausted, **cfg)
    ]


def update_rule_candidates_(
    rules_w_stats: List[Tuple[Rule, ChildRuleStats]],
    remaining_vocab: Set[ExamplePair],
    forget_lhs: Iterable[Rule] = None,
) -> None:
    """Update `rules_w_stats` in place (!)"""
    prune_idxs = []
    forget_lhs = forget_lhs or ()
    lhs_to_forget = {r.lhs for r in forget_lhs}
    parent_examples_by_tuple_id = {}
    for i, (rule, stats) in enumerate(rules_w_stats):
        if rule.lhs in lhs_to_forget:
            prune_idxs.append(i)
            continue
        _parent_examples = stats.get_parent_examples()  # type: Tuple[Set[ExamplePair], Set[ExamplePair]]
        updated_parent_examples = parent_examples_by_tuple_id.setdefault(
            id(_parent_examples),
            (_parent_examples[0] & remaining_vocab, _parent_examples[1] & remaining_vocab)
        )
        stats.update(remaining_vocab, updated_parent_examples)
    for idx in reversed(sorted(prune_idxs)):
        del rules_w_stats[idx]


def make_root_rule():
    return Rule(WILDCARD_CHAR, WILDCARD_CHAR)


def prime_rule(example: ExamplePair, *args, **kwargs) -> List[Rule]:
    try:
        rule = Rule(*_lhs_rhs_patterns(*example.overlap_masked), is_prime=True)
    except UncollapseableWildcardsException:
        return []
    if example.is_exhausted:
        raise TokenMaskExhaustedException(rule, example)
    else:
        return [rule]


def derive_new_rule(
    example: ExamplePair, rule: Rule, max_changes=1, *args, **kwargs
) -> List[Rule]:
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
    token_placeholder_spans = _captured_groups_as_spans(m)
    # Reverse engineered: token_mask that could have generated rule.
    token_mask = _mask_from_spans(token_placeholder_spans, len(token))
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
        if not new_lemma_mask:
            continue
        try:
            new_rule = Rule(
                *_lhs_rhs_patterns(
                    _apply_mask(token, new_token_mask), _apply_mask(lemma, new_lemma_mask)
                )
            )
        except UncollapseableWildcardsException:
            continue
        derived_rule_candidates.append((new_rule, new_token_mask))
    # If valid rules found by first strategy, select the rule whose lhs differs
    # the least from the lhs of the original rule.
    return [r for r, _ in derived_rule_candidates]

    #     # select least different
    #     sm = SequenceMatcher(None)
    #     sm.set_seq2(rule.lhs)
    #     most_sim_dr_i, max_sim = 0, 0
    #     for dr_i, (dr_rule, dr_token_mask) in enumerate(derived_rule_candidates):
    #         sm.set_seq1(dr_rule.lhs)
    #         similarity = sm.ratio()
    #         if similarity > max_sim:
    #             max_sim = similarity
    #             most_sim_dr_i = dr_i
    #     best_rule, underlying_token_mask = derived_rule_candidates[most_sim_dr_i]
    # # Fallback to prime rule lhs merged with token_mask according to lhs of current rule.
    # else:
    #     new_token_mask = underlying_token_mask = bytearray(
    #         map(op.and_, token_mask, example.token_mask)
    #     )
    #     new_lemma_mask = _infer_corresponding_lemma_mask(token, new_token_mask, lemma)
    #     if not new_lemma_mask:
    #         return None
    #     try:
    #         best_rule = Rule(
    #             *_lhs_rhs_patterns(
    #                 _apply_mask(token, new_token_mask), _apply_mask(lemma, new_lemma_mask)
    #             )
    #         )
    #     except UncollapseableWildcardsException:
    #         return None
    # if not any(underlying_token_mask):
    #     raise TokenMaskExhaustedException(best_rule, example)
    # else:
    #     return best_rule


def prime_rules(vocab: Vocab, rule: Rule = None, exhausted: Set[ExamplePair] = None, **kwargs):
    return _collect_rules(prime_rule, vocab, rule, exhausted)


def derived_rules(vocab: Vocab, rule: Rule, exhausted: Set[ExamplePair], max_changes=1, **kwargs):
    return _collect_rules(derive_new_rule, vocab, rule, exhausted, max_changes=max_changes)


def evaluate_rule(rule: Rule, vocab: Iterable[ExamplePair]) -> Tuple[Set[ExamplePair], ...]:
    correct, incorrect, not_matched = set(), set(), set()
    for x in vocab:
        yhat = rule.apply(x.token)
        if yhat is None:
            not_matched.add(x)
        elif yhat == x.lemma:
            correct.add(x)
        else:
            incorrect.add(x)
    return correct, incorrect, not_matched


# --------------------------------------------------------------------------------------


def _collect_rules(
    rule_gen_f, vocab: Vocab, rule: Rule = None, exhausted: Set[ExamplePair] = None, **kwargs
):
    seen_rules = set()
    prev_lhs = rule.lhs if rule else WILDCARD_CHAR
    for x in vocab:
        try:
            rules_ = rule_gen_f(x, rule, **kwargs)
        except TokenMaskExhaustedException:
            if exhausted:
                exhausted.add(x)
        else:
            for r in rules_:
                # print(r.lhs, r.rhs)
                if r and r.lhs != prev_lhs:
                    if r not in seen_rules:
                        seen_rules.add(r)
                        yield r


def _compute_overlap_masks(seq1, seq2) -> Tuple[bytearray, bytearray]:
    overlap = _sequence_overlap(seq1, seq2)
    seq1_mask = _mask_from_spans(((s_idx, s_idx + size) for s_idx, _, size in overlap), len(seq1))
    seq2_mask = _mask_from_spans(((s_idx, s_idx + size) for _, s_idx, size in overlap), len(seq2))
    return seq1_mask, seq2_mask


def _sequence_overlap(seq1: Sequence, seq2: Sequence):
    """Convenience function for computing overlap between two sequences.

    Returns:
         Tuples of the form (start idx in seq1, start idx in seq2, size/length).
    """
    diff = SequenceMatcher(None, seq1, seq2)
    # Drop last matching block, since this is always a dummy entry, with length=1.
    return diff.get_matching_blocks()[:-1]


def _mask_from_spans(spans: Iterable[Tuple[int, int]], seq_length: int) -> bytearray:
    mask = bytearray(seq_length)
    for s_i, e_i in spans:
        mask[s_i:e_i] = it.repeat(1, e_i - s_i)
    return mask


def _invert_mask(mask: bytearray) -> bytearray:
    return bytearray(1 ^ i for i in mask)  # `^` = XOR


def _apply_mask(s: str, mask: bytearray):
    return "".join("*" if m else x for x, m in zip(s, mask))


def _collapse_wildcards(s: str) -> str:
    return re.sub(f"{re.escape(WILDCARD_CHAR)}+", WILDCARD_CHAR, s)


def _rule_as_regexp(rule_lhs_pattern: str, rule_rhs_pattern: str):  # -> Callable[[str], str]:
    """
    Note:
        Rule pattern uses "*" to mean '__one__ or more characters'.
    """
    lhs_wildcards_n = sum(1 for s in rule_lhs_pattern if s == WILDCARD_CHAR)
    rhs_wildcards_n = sum(1 for s in rule_rhs_pattern if s == WILDCARD_CHAR)
    # print(rule_lhs_pattern, rule_rhs_pattern, lhs_wildcards_n, rhs_wildcards_n)
    assert lhs_wildcards_n == rhs_wildcards_n
    match_pattern = _regexify_matching_pattern(rule_lhs_pattern)
    sub_pattern = f"{rule_rhs_pattern}"
    for backref_n in range(1, lhs_wildcards_n + 1):
        sub_pattern, _ = re.subn(
            re.escape(WILDCARD_CHAR), r"\\g<{}>".format(backref_n), sub_pattern, count=1
        )
    return match_pattern, sub_pattern


def _regexify_matching_pattern(rule_pattern: str, wildcard_optional=False) -> str:
    """Regexifies pattern against which tokens will be matched (i.e. the left-
    hand side of the rule usually).
    """
    return rule_pattern.replace("*", f"(.{'+*'[wildcard_optional]})")


def _captured_groups_as_spans(m: ReMatch) -> Sequence[Tuple[int, int]]:
    return [m.span(g) for g in range(1, len(m.groups()) + 1)]


def _lhs_rhs_patterns(masked_token, masked_lemma) -> Tuple[str, str]:
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
        try:
            lemma_head, lemma_rest = _lemma.split(WILDCARD_CHAR * pl, maxsplit=1)
        except ValueError:
            raise UncollapseableWildcardsException
        _lemma_pattern.append(lemma_head)
        _lemma = lemma_rest
    _lemma_pattern.append(_lemma)
    lemma_pattern = WILDCARD_CHAR.join(_lemma_pattern)

    return token_pattern, lemma_pattern


def _is_subsumed_by(rule_pattern1, rule_pattern2):
    """
    Return `True` if rule_pattern2 is a generalization of rule_pattern1,
    or, equivalently, if rule_pattern1 is a more constrained version of
    rule_pattern2.
    """
    if rule_pattern1 == rule_pattern2:
        return False
    if re.match(_regexify_matching_pattern(rule_pattern2), rule_pattern1):
        return True
    else:
        return False


def _infer_corresponding_lemma_mask(token, token_mask, lemma) -> Optional[bytearray]:
    token_mask_inverted = _invert_mask(token_mask)
    # Re-mask token according to inverted modified character mask.
    # This unmasks the characters that would otherwise be masked; and masks
    # those characters which are normally visible in the rule pattern.
    _new_masked_token = _apply_mask(token, token_mask_inverted)
    # Transform _masked_token into usable _pattern.
    _pattern = _regexify_matching_pattern(
        _collapse_wildcards(_new_masked_token), wildcard_optional=True
    )
    # Match pattern against lemma.
    m = re.fullmatch(_pattern, lemma)
    if m:
        _new_lemma_mask = _invert_mask(_mask_from_spans(_captured_groups_as_spans(m), len(lemma)))
        return _new_lemma_mask
