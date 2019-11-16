import csv
import json
import time
from collections import defaultdict
import statistics
import itertools as it
import random
import logging

from trefwurd.tree import RuleTree, RuleTreeForest

elex_f = "/Users/adriaan/Dropbox/data/word_lemma_pos_elex.txt"
print(elex_f)

logging.basicConfig(level=logging.DEBUG)

random.seed(42)


def get_data():

    with open(elex_f) as fp:
        reader = csv.reader(fp, delimiter="\t")
        # examples_by_pos = defaultdict(lambda: defaultdict(list))
        examples_by_pos = defaultdict(dict)
        seen_tokens = {}
        for token, lemma, ud_pos, cgn_pos, cgn_freq in reader:
            cgn_freq = int(cgn_freq)
            if seen_tokens.get(token, -1) >= cgn_freq:
                continue
            else:
                examples_by_pos[ud_pos][token] = lemma
                seen_tokens[token] = cgn_freq
        # for ud_pos, data in examples_by_pos.items():
        #     for token in data.keys():
        #         data[token] = statistics.mode(data[token])

    return {
        ud_pos: random.sample(data.items(), k=len(data))
        for ud_pos, data in examples_by_pos.items()
    }


def ngrams(s, n=2):
    return (s[i:i + n] for i in range(len(s)))


def jaccard_similarity(s1, s2, n=2):
    s1_bigram_set = set(ngrams(s1, n=n))
    s2_bigram_set = set(ngrams(s2, n=n))
    return len(s1_bigram_set & s2_bigram_set) / len(s1_bigram_set | s2_bigram_set)


if __name__ == "__main__":
    lemmatization_data_by_pos = get_data()
    pos = "NOUN"

    size = 1000

    data = lemmatization_data_by_pos[pos][:size]
    # data = list(it.chain.from_iterable(lemmatization_data_by_pos.values()))
    # random.shuffle(data)
    data = data[:size]

    # inflected_examples = []
    # uninflected_examples = []
    #
    # for x in data:
    #     if x[0] == x[1]:
    #         uninflected_examples.append(x)
    #     else:
    #         inflected_examples.append(x)
    #
    # print(len(inflected_examples), len(uninflected_examples))
    # data = inflected_examples + random.sample(
    #     uninflected_examples, k=min(len(inflected_examples), len(uninflected_examples))
    # )
    random.shuffle(data)

    train_test_partition_idx = int(len(data) * 0.75)
    train_data, test_data = data[:train_test_partition_idx], data[train_test_partition_idx:]

    print(f"Examples in training set BEFORE filtering: {len(train_data)}")

    min_jaccard = 0.0
    print("Examples where token=lemma:", sum(1 for x in train_data if x[0] == x[1]))

    # train_data = [x for x in train_data if x[0] != x[1]]

    print(random.sample(train_data, k=20))

    print(f"Examples in training set AFTER filtering: {len(train_data)}")
    test_data = test_data[: min(len(train_data), len(test_data))]
    print(f"Examples in test set AFTER filtering: {len(train_data)}")

    x_train, y_train = zip(*train_data)
    x_test, y_test = zip(*test_data)

    # lemmatizer = RuleTree()
    # lemmatizer.fit(x_train, y_train)

    lemmatizer = RuleTreeForest(n_trees=3)  # , max_depth=51)
    lemmatizer.fit(x_train, y_train)

    # serialization test
    # tree = lemmatizer.trees[0]
    # tree = RuleTree.from_dict(tree.to_dict())
    # with open("/Users/adriaan/Projects/trefwurd/scripts/tmp/model.json", "wt") as fp:
    #     json.dump(trees_as_dict, fp, indent=2)

    train_data_test_start_time = time.time()
    # Accuracy on train data
    # lookup_table = noun_lemmatizer.lookup
    # lookup_table.update((l, l) for l in lemmas)
    correct = 0
    for tok, lemma in zip(x_train, y_train):
        lemma_pred = lemmatizer.predict(tok)
        lemma_pred = lemma_pred if lemma_pred is not None else tok
        if lemma_pred == lemma:
            correct += 1
            # print(f"{tok}. {lemma_pred} != {lemma}")
    print(f"Accuracy on training data: ({correct / len(train_data)})")
    print(
        f"Prediction took {time.time() - train_data_test_start_time}s on {len(train_data)} examples."
    )

    # ------------------------------------------------------------

    # lookup_table = noun_lemmatizer.lookup
    # lookup_table.update((l, l) for l in lemmas)
    # print("lookup table size:", len(lookup_table))
    print(f"Test cases: {len(x_test)}")
    correct = 0
    for tok, lemma in zip(x_test, y_test):
        lemma_pred = lemmatizer.predict(tok)
        lemma_pred = lemma_pred if lemma_pred is not None else tok
        if lemma_pred == lemma:
            correct += 1
    incorrect = len(x_test) - correct

    print(correct / len(x_test))

    if isinstance(lemmatizer, RuleTree):
        with open("/Users/adriaan/Projects/trefwurd/scripts/tmp/rules.txt", "wt") as fp:
            for n in it.chain([lemmatizer.root], lemmatizer.root.descendants):
                if n.rule:
                    fp.write(f"{n.rule.as_string()}\t({['derived', 'prime'][n.rule.is_prime]})\n")
    elif isinstance(lemmatizer, RuleTreeForest):
        for i, tree in enumerate(lemmatizer.trees):
            with open(f"/Users/adriaan/Projects/trefwurd/scripts/tmp/forest_tree{i}.rules.txt", "wt") as fp:
                print(tree.root.display(), file=fp)

