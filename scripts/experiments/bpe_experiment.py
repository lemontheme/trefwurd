from collections import Counter, defaultdict
import itertools as it
from typing import Dict, List, Iterable, Iterator, Tuple, Optional, Set
import random
import re
from operator import neg
import time

from tqdm import tqdm

from trefwurd.utils import read_lexical_data, character_ngrams


# Recreation of https://www.aclweb.org/anthology/P16-1162


def words2symbols(words: Iterable[str]) -> Iterator[str]:
    for word in words:
        yield " ".join(word)  # + " " + "__end__"


def prepare_word_corpus(words) -> List[str]:
    return list(words2symbols(words))


def count_characters(symbolized_words):
    return Counter(char for word in symbolized_words
                   for char in word.split())


class TokenLevelBPEEncoder:

    def __init__(self, n_merges=200, max_bp_len=None, verbose=True):
        self.n_merges = n_merges
        self.bpe_vocab = None
        self._vocab = None
        self._character_vocab = None

    def fit(self, words: Iterable[str]):
        # End result
        bpe_vocab = {}
        # Initialize vocab and character vocab
        ngrams_vocab = Counter()
        character_vocab = Counter()
        for word in tqdm(words):
            for char_ngram in character_ngrams(word, min_gram=1, max_gram=len(word)):
                if len(char_ngram) == 1:
                    character_vocab[char_ngram] += 1
                else:
                    ngrams_vocab[" ".join(char_ngram)] += 1  # 'word' -> 'wo' -> 'w o'

        # Sorted infrequent to frequent, which makes it more efficient to
        # pop off frequent bigrams.
        sorted_ngrams = [(bigram, freq) for bigram, freq in sorted(ngrams_vocab.items(),
                                                                   key=lambda x: x[1]) if freq > 2]
        # Incremental merging
        for _ in tqdm(range(self.n_merges)):

            best_ngram, freq = sorted_ngrams.pop()
            joined = best_ngram.replace(" ", "")
            # print(f"{best_ngram} => {joined}")
            # time.sleep(0.9)
            bpe_vocab[joined] = freq
            pair_re = re.compile(r"(?<!\S){}(?!\S)".format(best_ngram))
            merges = 0
            for i, (ngram, freq) in enumerate(sorted_ngrams):
                if best_ngram in ngram:
                    sorted_ngrams[i] = (pair_re.sub(joined, ngram), freq)
                    merges += 1

        bpe_vocab.update(character_vocab)
        bpes = ((bp, freq) for bp, freq in sorted(bpe_vocab.items(), key=lambda item: -item[1]))
        print(len(sorted_ngrams))
        print(sorted_ngrams[-100:])
        rest_vocab = [(s, -1) for ngram, _ in sorted_ngrams for s in ngram.split() if s not in bpe_vocab]
        print(rest_vocab)
        self.bpe_vocab = dict(it.chain(bpes, rest_vocab))

    def fit2(self, words: Iterable[str]):
        pairs_table = Counter()
        character_table = Counter()
        merged_pairs = {}

        for i, word in enumerate(tqdm(words)):
            for char_ngram in character_ngrams(word, min_gram=1, max_gram=len(word)):
                if len(char_ngram) == 1:
                    character_table[char_ngram] += 1
                else:
                    for pair in symbols_to_pairs(char_ngram):
                        pairs_table[pair] += 1
            if i % 10000 == 0:
                self.prune_pairs(pairs_table)

        for _ in tqdm(range(self.n_merges)):
            best_ngram, freq = pairs_table.most_common(1)[0]
            joined = best_ngram.replace(" ", "")

            del pairs_table[best_ngram]
            merged_pairs[joined] = freq

            print(f"{best_ngram} => {joined}")
            # time.sleep(0.9)
            pair_re = re.compile(r"(?<!\S){}(?!\S)".format(best_ngram))
            merges = 0

            updates = []
            for ngram, freq in pairs_table.items():
                if best_ngram in ngram:
                    updates.append((ngram, freq, pair_re.sub(joined, ngram)))
            for ngram, freq, new in updates:
                print(ngram, new, sep="==>")
                pairs_table[new] = freq + 1
                del pairs_table[ngram]
                merges += 1

        bpe_vocab = Counter(pair_comp for pair, freq in pairs_table.items()
                            for pair_comp in pair.split() if pair_comp in merged_pairs)
        bpe_vocab.update(merged_pairs)
        bpe_vocab.update(character_table)
        self.bpe_vocab = bpe_vocab

    @staticmethod
    def prune_pairs(pairs_table):
        keys_to_trim = [k for k, v in pairs_table.items() if v <= 3]
        for k in keys_to_trim:
            del pairs_table[k]



    # def _update_vocab(self, pair) -> None:
    #     """MERGE step."""
    #     corpus = self._corpus
    #     for idx in self._pairs2corpus_idxs.get(pair, []):
    #         old = corpus[idx]
    #         new = pair_re.sub(joined, old)
    #         assert old != new
    #         corpus[idx] = new

    # def encode(self, word):
    #     pass

#     def _update_bigram_counts(self) -> None:
#
#         _bigrams2wordidxs = self._pairs2corpus_idxs
#
#         per_word_bigram_counts = Counter()  # used and reset within for-loop
#
#         words = self._corpus
#
#         for idx, word in enumerate(words):
#             symbols = word.split()
#             symbol_bigrams = (" ".join(symbols[start_idx:start_idx+2])
#                               for start_idx in range(len(symbols) - 1))
#
#             per_word_bigram_counts.update(symbol_bigrams)
#             for bigram in per_word_bigram_counts:
#                 _bigrams2wordidxs[bigram].append(idx)
#
#             symbol_pair_counts.update(per_word_bigram_counts)
#
#             per_word_bigram_counts.clear()  # !
#
#         return symbol_pair_counts, _bigrams2wordidxs
#
#     def save(self):
#         pass
#
#     @classmethod
#     def load(cls, f):
#         pass
#
#
# def learn_bpe_vocab(words, max_iters=10, max_len=3):
#
#     corpus = prepare_word_corpus(words)
#
#     print("words in corpus: {}".format(len(corpus)))
#     character_vocab = count_characters(corpus)
#     bpe_vocab = Counter()
#
#     pair_counts = Counter()
#
#     for _ in tqdm(range(max_iters)):
#         pair_counts, pair2word_idxs = count_symbol_pairs(corpus, pair_counts)
#         top_n = pair_counts.most_common(4)
#         if not pair_counts:
#             break
#         best_pair, _ = top_n[0]
#         # print(best_pair)
#         merge_candidate_idxs = pair2word_idxs[best_pair]
#         # print(len(merge_candidate_idxs))
#         update_corpus(best_pair,
#                       corpus,
#                       merge_candidate_idxs)
#
#         bpe_vocab = Counter(symbol for w in corpus for symbol in w.split())
#         print("bpe vocab:", len(bpe_vocab), "; words to be merged:", len(merge_candidate_idxs), "; pair:", best_pair)
#
#     for char, freq in character_vocab.items():
#         bpe_vocab[char] = freq
#
#     return bpe_vocab, corpus

# def main():
#
#     lexical_data_f = "data/processed/word_lemma_pos_elex.txt"
#     data = read_lexical_data(lexical_data_f)
#
#     words = it.chain(it.islice(data.get("noun"), None),
#                      it.islice(data.get("verb"), None))
#
#     # vocab = learn_bpe_vocab(nouns)
#     # print(vocab.most_common(1000))
#     # pairs = count_symbol_pairs(words2symbols(nouns))
#     # print(pairs.most_common(200))
#     # print(count_characters(words2symbols(nouns)))
#     bpe_vocab, corpus = learn_bpe_vocab(words, max_iters=1000)
#
#     with open("scripts/tmp/bpe_vocab.txt", "wt") as fp:
#         fp.write("\n".join("\t".join(map(str, x)) for x in bpe_vocab.most_common()))
#     with open("scripts/tmp/bpe_corpus.txt", "wt") as fp:
#         fp.write("\n".join(corpus))


def test():

    def get_pair_statistics(vocab):
        """Count frequency of all symbol pairs, and create index"""

        # data structure of pair frequencies
        stats = defaultdict(int)

        #index from pairs to words
        indices = defaultdict(lambda: defaultdict(int))

        for i, (word, freq) in enumerate(vocab):
            prev_char = word[0]
            for char in word[1:]:
                stats[prev_char, char] += freq
                indices[prev_char, char][i] += 1
                prev_char = char

        return stats, indices

    lexical_data_f = "data/processed/word_lemma_pos_elex.txt"
    data = read_lexical_data(lexical_data_f)
    words = it.chain(it.islice(data.get("noun"), 20000))
    vocab = dict(((tuple(x[:-1])+(x[-1]+'</w>',), 1) for x in words))
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    stats, indices = get_pair_statistics(sorted_vocab)
    print(indices)


def main():

    lexical_data_f = "data/processed/word_lemma_pos_elex.txt"
    data = read_lexical_data(lexical_data_f)
    words = it.chain(it.islice(data.get("noun"), 20000))
    # words = list(words)
    # print(len(words))
    # words = random.sample(words, 10000)

    bpe = TokenLevelBPEEncoder(n_merges=1000)
    bpe.fit2(words)
    bpe_vocab = bpe.bpe_vocab

    with open("scripts/tmp/bpe_vocab.txt", "wt") as fp:
        fp.write("\n".join("\t".join(map(str, x))
                           for x in sorted(bpe_vocab.items(), key=lambda t: t[1], reverse=True)))
    # with open("scripts/tmp/bpe_corpus.txt", "wt") as fp:
    #     fp.write("\n".join(corpus))


def symbols_to_pairs(symbols_string: str) -> Iterator[str]:
    for i in range(1, len(symbols_string)):
        yield "{} {}".format(symbols_string[:i], symbols_string[i:])


# ----------------------------------------------------------

def count_bigrams(word):
    return Counter(bigrams(word))


def counts_delta(old_counts, new_counts):
    delta = {}
    for count_key in old_counts.keys() | new_counts.keys():
        old_count = old_counts.get(count_key)
        new_count = new_counts.get(count_key)
        if old_count is None:
            delta[count_key] = new_counts[count_key]
        elif new_count is None:
            delta[count_key] = neg(old_count)
        else:
            delta[count_key] = new_count - old_count
    return delta


def word2symbol(word: str) -> str:
    return " ".join(word)


def prepare_vocab(tokens: Iterable[str]):
    _vocab = Counter(tokens)
    return {word2symbol(word): freq for word, freq in _vocab.items()}


VocabIndices = Dict[str, Set[int]]
PairStats = Dict[str, int]


def bigrams(seq):
    for start_idx in range(len(seq) - 1):
        yield " ".join(seq[start_idx:start_idx+2])


def get_pair_stats(vocab) -> Tuple[PairStats, VocabIndices]:

    stats = defaultdict(int)
    indices = defaultdict(set)

    for word_idx, (word, freq) in enumerate(vocab):
        for pair in bigrams(word.split()):
            stats[pair] += freq
            indices[pair].add(word_idx)

    return stats, indices


# ! Side-effectual (would make a good method)
def merge_and_update_in_place(pair_to_merge: str,
                              vocab: List[Tuple[str, int]],
                              stats: PairStats,
                              indices: VocabIndices) -> None:

    merge_symbol = pair_to_merge.replace(" ", "")
    pair_re = re.compile(r"(?<!\S){}(?!\S)".format(pair_to_merge))

    word_idxs = list(indices[pair_to_merge])
    # print("words to merge:", len(word_idxs))

    for word_idx in word_idxs:

        old_word, freq = vocab[word_idx]
        new_word = pair_re.sub(merge_symbol, old_word)

        vocab[word_idx] = new_word, freq

        bigram_stats_delta = counts_delta(
            count_bigrams(old_word.split()), count_bigrams(new_word.split()))

        for bigram, delta_value in bigram_stats_delta.items():
            if delta_value == 0:
                pass
            else:
                if delta_value < 0:
                    indices[bigram].discard(word_idx)
                else:
                    indices[bigram].add(word_idx)
                stats[bigram] += delta_value * freq  # effect: decrease value

    stats[pair_to_merge] = -1
    del indices[pair_to_merge]


def learn_bpe(tokens, n_merge_ops=100, min_pair_freq=2) -> Iterator[str]:

    vocab: List[Tuple[str, int]]
    stats: PairStats
    indices: VocabIndices

    min_pair_freq = max(min_pair_freq, 0)  # ensure that value >= 0

    vocab = sorted(prepare_vocab(tokens).items(),
                   key=lambda x: x[1], reverse=True)

    stats, indices = get_pair_stats(vocab)

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

        merge_and_update_in_place(most_freq_pair, vocab, stats, indices)

        if i % 100 == 0:
            print("Pruning stats...")
            # OR will evaluate to right-side ONLY when i = 0 (i.e., at the first
            # iteration)
            threshold = int(pair_freq * i/(i+10000) or min_pair_freq)
            print("Pruning threshold: {}".format(threshold))
            prune_stats(stats, threshold)

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


def main_1():
    lexical_data_f = "data/processed/word_lemma_pos_elex.txt"
    data = read_lexical_data(lexical_data_f)
    words = list(it.chain(it.islice(data.get("noun"), None)))
    # words = random.sample(words, 10000)
    bpes = learn_bpe(words, 10000)
    with open("scripts/tmp/bpe_vocab.txt", "wt") as fp:
        fp.write("\n".join(bpes))




if __name__ == '__main__':
    # print(list(character_ngrams("hellothere")))
    # main()
    # test()
    main_1()






