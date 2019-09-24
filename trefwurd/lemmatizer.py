from typing import List, Tuple
import logging


logger = logging.getLogger(__name__)


class TrefwurdLemmatizer:

    def fit(self, wordclass_tokens: List[Tuple[str, str]], lemmas: List[str], max_iters=20):
        ...

    def transform(self, tokens):
        ...

    @classmethod
    def load(cls, f):
        return ...

    def _as_spacy_lemmatizer(self) -> "SpacyLemmatizer":
        _ = self
        return SpacyLemmatizer()


class SpacyLemmatizer:

    def __init__(self, index=None, exceptions=None, rules=None, lookup=None):
        ...


def create_rule(full_form, lemma, current_rule_length):
    raise NotImplementedError


def apply_rule(rule, full_form):
    raise NotImplementedError






