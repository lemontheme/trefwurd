"""Byte-pair encoding"""

import itertools as it
import re

from collections import Counter, defaultdict
from operator import neg
from collections.abc import Sequence as SequenceABC
from typing import Dict, List, Iterable, Iterator, Tuple, Set, Sequence, Union

from tqdm import tqdm


class BpeTransformer:
    """Subtokenizes pre-tokenized documents into byte-pairs."""

    bpe_vocab: Dict[str, int]

    def __init__(self, merges=1000, min_freq=2, verbose=False):
        self.merges = merges
        self.min_freq = min_freq

    def fit(self, x_: Iterable[Union[Sequence[str], str]]) -> None:
        """ Learn BPE vocab.
        Args:
            x_: pretokenized documents OR tokens
        Returns: None
        """
        x_ = iter(x_)
        x_first, x_rest = next(x_), x_
        if isinstance(x_first, str):
            tokens_iter = it.chain([x_first], x_rest)
        elif isinstance(x_first, SequenceABC):
            tokens_iter = (token for token in it.chain([x_first], x_rest))
        else:
            raise ValueError

        self.bpe_vocab = {}
        for idx, bpe in enumerate(learn_bpe(tokens_iter, self.merges, self.min_freq)):
            self.bpe_vocab[bpe] = idx

    def transform(self, words: Iterable[Sequence[str]]) -> Iterable[Sequence[str]]:
        raise NotImplementedError

    @classmethod
    def load(cls, f):
        raise NotImplementedError

    @classmethod
    def save(cls):
        raise NotImplementedError


def bigrams_as_strings(seq) -> Iterator[str]:
    for start_idx in range(len(seq) - 1):
        yield " ".join(seq[start_idx:start_idx+2])


def bigram_counts(word) -> Counter:
    return Counter(bigrams_as_strings(word))


def bigram_counts_deltas(old_counts, new_counts):
    deltas = {}
    for count_key in old_counts.keys() | new_counts.keys():
        old_count = old_counts.get(count_key)
        new_count = new_counts.get(count_key)
        if old_count is None:
            deltas[count_key] = new_counts[count_key]
        elif new_count is None:
            deltas[count_key] = neg(old_count)
        else:
            deltas[count_key] = new_count - old_count
    return deltas


def word2symbol(word: str) -> str:
    return " ".join(word)


def prepare_vocab(tokens: Iterable[str]):
    _vocab = Counter(tokens)
    return {word2symbol(word): freq for word, freq in _vocab.items()}


VocabIndices = Dict[str, Set[int]]
PairStats = Dict[str, int]


def get_pair_stats(vocab) -> Tuple[PairStats, VocabIndices]:

    stats = defaultdict(int)
    indices = defaultdict(set)

    for word_idx, (word, freq) in enumerate(vocab):
        for pair in bigrams_as_strings(word.split()):
            stats[pair] += freq
            indices[pair].add(word_idx)

    return stats, indices


# ! Side-effectual
def merge_bigram_and_update_stats_in_place(pair_to_merge: str,
                                           vocab: List[Tuple[str, int]],
                                           stats: PairStats,
                                           indices: VocabIndices) -> None:

    merge_symbol = pair_to_merge.replace(" ", "")
    pair_re = re.compile(r"(?<!\S){}(?!\S)".format(pair_to_merge))

    word_idxs = list(indices[pair_to_merge])

    for word_idx in word_idxs:

        old_word, freq = vocab[word_idx]
        new_word = pair_re.sub(merge_symbol, old_word)

        vocab[word_idx] = new_word, freq

        bigram_stats_delta = bigram_counts_deltas(
            bigram_counts(old_word.split()), bigram_counts(new_word.split()))

        for bigram, delta_value in bigram_stats_delta.items():
            if delta_value == 0:
                pass
            else:
                if delta_value < 0:
                    indices[bigram].discard(word_idx)
                else:
                    indices[bigram].add(word_idx)
                stats[bigram] += delta_value * freq  # effect: increment or decrement value

    stats[pair_to_merge] = -1
    del indices[pair_to_merge]


def learn_bpe(tokens: Iterable[str], n_merge_ops: int, min_pair_freq=2) -> Iterator[str]:

    vocab: List[Tuple[str, int]]
    stats: PairStats
    indices: VocabIndices

    min_pair_freq = max(min_pair_freq, 0)  # ensure that value >= 0

    vocab = sorted(prepare_vocab(tokens).items(),
                   key=lambda x: x[1], reverse=True)

    stats, indices = get_pair_stats(vocab)

    first_max_freq = max(stats.values())
    threshold = first_max_freq / 10000
    prev_freq = None

    for i in tqdm(range(n_merge_ops)):

        try:
            most_freq_pair = max(stats, key=lambda x: (stats[x], neg(len(x[0]))))
        except ValueError:
            print("Run out of pairs in `stats`. (Iteration: {})".format(i))
            break

        pair_freq = stats[most_freq_pair]
        if pair_freq < min_pair_freq:
            print("Hit minimum pair frequency. Exiting loop.")
            break

        merge_bigram_and_update_stats_in_place(most_freq_pair, vocab, stats, indices)

        if i % 100 == 0:
            print("Pruning stats...")
            # OR will evaluate to right-side ONLY when i = 0 (i.e., at the first
            # iteration)
            # e.g. 9 * ((5 + (8 - 5) * (1- w)) / 8)
            if i == 0:
                threshold = threshold
            else:
                w = 0.0001 #  * i/(i+10000)
                scale = (pair_freq + (prev_freq - pair_freq) * (1 - w)) / prev_freq
                threshold = max(int(threshold * scale), min_pair_freq)
                # threshold = prev_threshold *
            print("Pruning threshold: {} (<= {}).".format(threshold, pair_freq))
            prune_stats(stats, threshold)

        prev_freq = pair_freq

        yield most_freq_pair


def prune_stats(stats, threshold: int):
    """Prune statistics dict for efficiency of max()
    The frequency of a symbol pair never increases, so pruning is generally safe
    (until we the most frequent pair is less frequent than a pair we previously pruned)
    big_stats keeps full statistics for when we need to access pruned items
    """
    for item,freq in list(stats.items()):
        if freq < threshold:
            del stats[item]


# ON-SITE TESTING

if __name__ == '__main__':
    from .utils import read_lexical_data
    lexical_data_f = "data/processed/word_lemma_pos_elex.txt"
    data = read_lexical_data(lexical_data_f)
    words = list(it.chain(it.islice(data.get("noun"), None)))
    encoder = BpeTransformer()
    encoder.fit(words)
    print(encoder.bpe_vocab)


