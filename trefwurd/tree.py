import itertools as it
import multiprocessing
import operator as op
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import (
    Tuple,
    Sequence,
    Optional,
    Set,
    Callable,
    Union,
    Iterator,
    List,
    Any,
    Dict,
    Collection,
    Iterable,
)
import logging
import math
import statistics

from .rules import Rule, ExamplePair, ChildRuleStats, make_rule_candidates, update_rule_candidates_
from .utils import binary_entropy

logger = logging.getLogger(__name__)


class BinaryNode:

    parent: Optional["BinaryNode"] = None
    left: Optional["BinaryNode"] = None
    right: Optional["BinaryNode"] = None
    id_: Optional[int]

    def attach_to(self, node, branch: str = "left", as_: str = "child"):
        if branch not in ("left", "right") or as_ not in ("child", "parent"):
            raise ValueError
        if node == self:
            raise ValueError("Nodes cannot attach to themselves, since this introduces cyclicity.")
        if as_ == "child":
            self.parent = node
            if branch == "left":
                node.left = self
            else:
                node.right = self
        else:
            node.parent = self
            if branch == "left":
                self.left = node
            else:
                self.right = node

    def detach_from(self, node):
        if node is self.parent:
            self.parent = None
            for attr in ("left", "right"):
                if self is getattr(node, attr):
                    setattr(node, attr, None)
        else:
            raise ValueError

    @property
    def ancestors(self) -> List["BinaryNode"]:
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

    @property
    def descendants(self) -> List["BinaryNode"]:
        stack = [self]
        result = []
        while stack:
            node = stack.pop()
            if node is not self:
                result.append(node)
            stack.extend(n for n in (node.left, node.right) if n)
        return result

    @property
    def leaves(self) -> List["BinaryNode"]:
        return [node for node in self.descendants if node.is_leaf]

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    def display(self, _indent=None):
        indent = _indent or 0
        s = [f"{' ' * indent}{self}"]
        child_indent = indent + 2
        if self.left:
            s.append(f"{self.left.display(child_indent)}")
        if self.right:
            s.append(f"{self.right.display(child_indent)}")
        return "\n".join(s)


class RuleNode(BinaryNode):
    """
    left (STRNode): matched branch
    right (STRNode): non-matched branch
    score: value returned by scorer.
    """
    rule: Rule
    samples: Optional[Tuple[float, float]]
    impurity: float
    probs: Tuple[float, float]
    _meta: Optional[Dict[str, Any]]

    def __init__(self, samples: Tuple[int, int] = None, rule: Rule = None, id_=None, meta=None):
        self.rule = rule
        self.id_ = hash(self) if id_ is None else id_
        self._samples = samples
        self._meta = meta
        self._probs = None
        self._impurity = None
        self._m_estimate = None
        super().__init__()

    @property
    def meta(self):
        if not self._meta:
            self._meta = {}
        return self._meta

    def __hash__(self):
        return id(self) ^ hash(self.rule)

    @property
    def probs(self):
        if not self._probs and self.samples:
            p, n = self.samples
            dnm = max(p + n, 1)
            self._probs = p / dnm, n / dnm
        return self._probs

    @property
    def m_estimate(self):
        if not self._m_estimate and self.samples:
            self._m_estimate = node_m_estimate(self)
        return self._m_estimate

    @property
    def impurity(self):
        if not self._impurity and self.samples:
            self._impurity = binary_entropy(self.samples)
        return self._impurity

    @property
    def samples(self) -> Optional[Tuple[int, int]]:
        if self._samples:
            return self._samples
        else:
            return None

    @samples.setter
    def samples(self, val: Tuple[int, int]) -> None:
        self._samples = val
        self._impurity = None
        self._probs = None
        self._m_estimate = None

    def __repr__(self):
        rule_str = self.rule.to_string() if self.rule else None
        parent_rule_str = (
            self.parent.rule.to_string() if self.parent and self.parent.rule else None
        )
        return f"{type(self).__name__}(id_={self.id_}, rule={rule_str}, parent={parent_rule_str}, samples={self._samples})"


class RuleTree:
    def __init__(self, root=None, **cfg):
        self.root = root  # type: Optional[RuleNode]
        self.cfg = cfg  # type: Dict[str, Any]

    def _find_and_apply_matching_rule(self, s: str) -> Tuple[Optional[str], Optional[RuleNode]]:
        """Search binary tree for an STRNode whose rule matches `s` and apply rule.

        Args:
            node: The root of the (sub)tree to be searched.
            s: The token string.

        Returns:
             (result of applying associated rule, the rule node itself)
        """
        node = self.root
        best_match, matching_rule = None, None
        while node:  # terminates if node.left or node.right is None
            rule = node.rule
            if not rule:
                break
            m = rule.match(s)
            if m:
                best_match, matching_rule = m, rule
                node = node.left  # left child represents matching branch
            else:
                node = node.right
        result = matching_rule.transform_match(best_match) if best_match else None
        if random.random() > 0.99:
            logger.debug(f"Found and applied: {s} -> {matching_rule} -> {str(result)}")
        return result, node

    def fit(self, x: Sequence[str], y: Sequence[str]) -> "RuleTree":

        if not len(x) == len(y):
            raise ValueError("Seqs `x` (tokens) and `y` (lemmas) have unequal lengths.")
        data = list(ExamplePair(x, y) for x, y in zip(x, y))

        builder = Builder(self)
        builder.build(data)

        return self

    def predict(self, s: str) -> str:
        try:
            result, node = self._find_and_apply_matching_rule(s)
            return result
        except AttributeError:
            raise RuntimeError("predict() called on untrained RuleTree.")

    def evaluate(self, x, y, return_lookups=False, _blocked_nodes=None):
        correct, incorrect = 0, 0
        root = self.root
        missed_lookup = {} if return_lookups else None
        incorrect_lookup = {} if return_lookups else None
        for x, y in zip(x, y):
            pred, _, _ = _find_and_apply_modified(root, x, blocked_nodes=_blocked_nodes)
            if pred is None:
                if missed_lookup:
                    missed_lookup[x] = y
                if x == y:
                    correct += 1
            else:
                if pred == y:
                    correct += 1
                else:
                    incorrect += 1
                    if incorrect_lookup:
                        incorrect_lookup[x] = y
        return {
            "accuracy": correct / max(correct + incorrect, 1),
            **(
                {"lookups": {"missed": missed_lookup, "incorrect": incorrect_lookup}}
                if return_lookups
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RuleTree":
        id2node = {}
        nodes_data = data["nodes"]
        for x in nodes_data:
            id_ = x["id_"]
            rule_string = x["rule"]
            rule = Rule.from_string(rule_string) if rule_string else None
            id2node[id_] = RuleNode(id_=id_, rule=rule, samples=x["samples"])
        for x in nodes_data:
            node = id2node[x["id_"]]
            left_child_id, right_child_id = x["left_child"], x["right_child"]
            if left_child_id is not None:
                id2node[left_child_id].attach_to(node, as_="child", branch="left")
            if right_child_id is not None:
                id2node[right_child_id].attach_to(node, as_="child", branch="right")
        parentless = []
        for n in id2node.values():
            if n.parent is None:
                parentless.append(n)
        if len(parentless) > 1 or not parentless:
            raise RuntimeError
        root = parentless.pop()
        return RuleTree(root, **data.get("cfg", {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cfg": self.cfg,
            "nodes": [
                {
                    "id_": n.id_,
                    "rule": n.rule.to_string() if n.rule is not None else None,
                    "samples": n.samples,
                    "parent": getattr(n.parent, "id_", None),
                    "left_child": getattr(n.left, "id_", None),
                    "right_child": getattr(n.right, "id_", None),
                }
                for n in it.chain([self.root], self.root.descendants)
            ]
        }


def _find_and_apply_modified(
    tree: RuleNode, s: str, blocked_nodes: Set[RuleNode] = None
) -> Tuple[str, RuleNode, float]:
    blocked_nodes = blocked_nodes or set()
    node = tree
    best_match, matching_rule, matching_node = None, None, None
    while node:  # terminates if node.left or node.right is None
        if node in blocked_nodes:
            break
        rule = node.rule
        if not rule:
            break
        m = rule.match(s)
        if m:
            best_match, matching_rule, matching_node = m, rule, node
            node = node.left  # left child represents matching branch
        else:
            node = node.right
    result = matching_rule.transform_match(best_match) if best_match else None
    m_estimate = matching_node.left.m_estimate if matching_node else node.m_estimate
    return result, (matching_node or node), m_estimate


class RuleTreeForest:

    def __init__(self, **cfg):
        self.n_trees = cfg.get("n_trees", 3)
        self.cfg = cfg
        self.trees = []

    def fit(self, x: Sequence[str], y: Sequence[str]) -> "RuleTreeForest":

        if not len(x) == len(y):
            raise ValueError("Seqs `x` (tokens) and `y` (lemmas) have unequal lengths.")

        data = list(ExamplePair(x, y) for x, y in zip(x, y))

        bag_size = int(math.sqrt(1 / self.n_trees) * len(data))

        workers = multiprocessing.cpu_count() // 2 - 1
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(self.n_trees):
                train_fold = random.choices(data, k=bag_size)
                f = executor.submit(_build_tree, train_fold, self.cfg)
                futures.append(f)
            for f in as_completed(futures):
                self.trees.append(f.result())
        return self

    def predict(self, x):
        votes = defaultdict(float)
        for tree in self.trees:
            pred, _, m_estimate = _find_and_apply_modified(tree.root, x)
            votes[pred] += 2 ** m_estimate
        try:
            pred = max(votes.items(), key=op.itemgetter(1))[0]
        except (IndexError, ValueError, statistics.StatisticsError):
            pred = None
        if random.random() > 0.98:
            logger.debug(f"Found and applied: {x} -> [R] -> {pred}, with votes = {votes}")
        return pred

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cfg": self.cfg,
            "trees": [tree.to_dict() for tree in self.trees]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RuleTreeForest":
        forest = RuleTreeForest(**data.get("cfg", {}))
        for tree_data in data.get("trees"):
            tree = RuleTree.from_dict(tree_data)
            forest.trees.append(tree)
        return forest


def _build_tree(examples, cfg) -> RuleTree:
    tree = RuleTree(**cfg)
    builder = Builder(tree)
    builder.build(examples)
    return tree


# class Pruner:
#     def __init__(self, tree, **cfg):
#         self.tree = tree  # type: RuleTree
#         self.cfg = cfg
#
#     def find_best_alpha(self, dev_data=None):
#         root = self.tree.root
#         nodes_pruned = set()
#         alphas = []
#         val_accs = []
#         for subtree_nodes_pruned, n_leaves, complexity_score in self._compute_subtrees():
#             nodes_pruned.update(subtree_nodes_pruned)
#             alphas.append(complexity_score)
#
#             correct, incorrect = 0, 0
#             for x in dev_data:
#                 pred = (
#                     _find_and_apply_modified(root, x.token, blocked_nodes=nodes_pruned) or x.token
#                 )
#                 if pred == x.lemma:
#                     correct += 1
#                 else:
#                     incorrect += 1
#             val_accuracy = correct / max(correct + incorrect, 1)
#             val_accs.append(val_accuracy)
#
#             print(complexity_score, n_leaves, val_accuracy)
#
#         max_val_acc_idx = argmax_i(val_accs)
#
#         print(
#             f"Max accuracy on validation data: {val_accs[max_val_acc_idx]} (index={max_val_acc_idx}). (alpha score = {alphas[max_val_acc_idx]})"
#         )
#
#         return alphas[max_val_acc_idx]
#
#     def prune(self, alpha) -> None:
#         nodes_pruned = set()
#         for subtree_nodes_pruned, n_leaves, complexity_score in self._compute_subtrees():
#             nodes_pruned.update(subtree_nodes_pruned)
#             if complexity_score > alpha:
#                 print(f"Pruning at {complexity_score}")
#                 break
#             print("Computing subtrees...")
#         for n in nodes_pruned:
#             try:
#                 n.detach_from(n.parent)
#             except ValueError:
#                 pass
#
#     def _compute_subtrees(self) -> Iterator[Tuple[Collection[RuleNode], int, float]]:
#         nodes = self.tree.root.descendants
#         total_samples = sum(self.tree.root.samples)
#         leaves_frontier = {n for n in nodes if n.is_leaf}
#         pruned_nodes = set()
#
#         for _ in range(len(nodes)):
#             scored_pruning_candidates = []
#             n: Union[BinaryNode, RuleNode]
#             for n in nodes:
#                 if n in pruned_nodes or n.is_leaf:
#                     continue
#                 subtree_leaves = [d for d in n.descendants if d in leaves_frontier]
#                 n_leaves_pruned = len(subtree_leaves)
#                 R_t = n.probs[1] * sum(n.samples) / total_samples
#                 # noinspection PyUnresolvedReferences
#                 R_T_t = sum(n_.probs[1] * sum(n_.samples) / total_samples for n_ in subtree_leaves)
#                 alpha_score = (R_t - R_T_t) / max(n_leaves_pruned - 1, 1)
#                 # Including No. leaves pruned so that, in the event of a tie, subtrees with fewer nodes are preferred.
#                 scored_pruning_candidates.append((n, (alpha_score, n_leaves_pruned)))
#
#             if not scored_pruning_candidates:
#                 break
#
#             best_node_to_prune, (best_alpha_score, n_leaves) = min(
#                 scored_pruning_candidates, key=lambda c: c[1]
#             )
#
#             pruned_subtree_descendants = best_node_to_prune.descendants
#
#             nodes_pruned_in_this_step = [best_node_to_prune] + pruned_subtree_descendants
#             pruned_nodes.update(nodes_pruned_in_this_step)
#
#             pruned_nodes.update(nodes_pruned_in_this_step)
#
#             leaves_frontier.difference_update(pruned_subtree_descendants)
#             leaves_frontier.add(best_node_to_prune)
#
#             yield nodes_pruned_in_this_step, n_leaves, best_alpha_score
#
#     def _cost_complexity(self, tree):
#         raise NotImplementedError


class Builder:
    def __init__(self, tree: RuleTree):
        self.tree = tree
        self.cfg = self.tree.cfg

    def build(self, data: Sequence[ExamplePair]) -> None:
        tree = self._grow_tree(data)
        self.tree.root = tree
        # tree = self._finalize(tree)

    def _grow_tree(self, data: Sequence[ExamplePair]):

        max_nodes = self.cfg.get("max_nodes", 10e9)
        max_depth = self.cfg.get("max_depth", 10e9)

        print(max_nodes, max_depth)

        node_id_generator: Iterator[int] = it.count(0)

        _corr: Set[ExamplePair]
        _incorr: Set[ExamplePair]

        _corr = {x for x in data if x.token == x.lemma}
        _incorr = {x for x in data if x.token != x.lemma}

        root_node = RuleNode(id_=next(node_id_generator))
        root_node.samples = (len(_corr), len(_incorr))

        root_node.meta["examples"] = (frozenset(_corr), frozenset(_incorr))
        prime_rule_candidates = make_rule_candidates(_corr, _incorr)
        root_node.meta["candidate_child_rules"] = prime_rule_candidates

        active_nodes: List[RuleNode] = [root_node]
        final_nodes: List[RuleNode] = []

        _max_nodes_exceeded = (
            lambda: len(active_nodes) + len(final_nodes) > max_nodes
            if max_nodes
            else lambda: False
        )

        while active_nodes:

            logger.debug(f"Nodes: {len(active_nodes) + len(final_nodes)}")

            if _max_nodes_exceeded():
                logger.debug("Hit maximum number of nodes")
                break

            active_nodes.sort(key=lambda node: node.probs[1])

            node = active_nodes.pop()

            if node.depth >= max_depth:
                logger.debug(f"Hit max. node node depth d={max_depth}")
                final_nodes.append(node)
                continue

            if node.samples[1] == 0:
                logger.debug(f"All examples lemmatized correctly.")
                final_nodes.append(node)
                continue

            print(f"splitting node (samples={node.samples}): {node}")

            candidates: List[Tuple[Rule, ChildRuleStats]]
            update_factory: Callable
            candidates = node.meta.pop("candidate_child_rules")
            # filter candidates
            scores = [cst_score(stats) for _, stats in candidates]
            scored_candidates = [
                (rule, stats, score) for (rule, stats), score in zip(candidates, scores) if score
            ]
            scored_candidates.sort(key=op.itemgetter(2))

            if not scored_candidates:
                logger.debug(f"Node {node.id_} ran out of children.")
                final_nodes.append(node)
                continue

            print("candidates:", len(candidates))

            best_rule, best_rule_stats, best_rule_score = scored_candidates.pop()
            best_rule_corr, best_rule_incorr, best_rule_not_matched = best_rule_stats.examples

            candidates = [(rule, stats) for rule, stats, _ in scored_candidates]
            print(
                f"best rule: {best_rule}; matched: {best_rule_stats.counts.m_total}; "
                f"is {('derived', 'prime')[best_rule.is_prime]}; "
                f"score = {best_rule_score}"
            )

            node.rule = best_rule

            # left branch (matches rule) ----------------------------------------

            left_branch_children = make_rule_candidates(
                best_rule_corr, best_rule_incorr, rule=best_rule, max_changes=1
            )

            print(random.sample(left_branch_children, k=min(5, len(left_branch_children))))

            left_child_node = RuleNode(
                samples=(best_rule_stats.counts.c_total, best_rule_stats.counts.i_total),
                id_=next(node_id_generator),
                meta={
                    "examples": (frozenset(best_rule_corr), frozenset(best_rule_incorr)),
                    "candidate_child_rules": left_branch_children,
                },
            )

            left_child_node.attach_to(node, branch="left", as_="child")

            active_nodes.append(left_child_node)

            # right branch (not matched) ----------------------------------------

            not_matched_corr, not_matched_incorr = (
                frozenset(best_rule_not_matched.intersection(s)) for s in node.meta["examples"]
            )

            right_child_node = RuleNode(
                samples=(best_rule_stats.counts.nmc, best_rule_stats.counts.nmi),
                id_=next(node_id_generator),
                meta={"examples": (not_matched_corr, not_matched_incorr)},
            )

            right_child_node.attach_to(node, branch="right", as_="child")

            if best_rule_stats.counts.nm_total > 0:
                logger.debug(
                    f"Next right node from {len(best_rule_stats.examples.not_matched)} unmatched examples"
                )

                update_rule_candidates_(
                    candidates,
                    remaining_vocab=best_rule_stats.examples.not_matched,
                    forget_lhs=[best_rule],
                )

                right_branch_children = (
                    candidates
                )  # modified in place by `update_rule_candidates_()`

                right_child_node.meta["candidate_child_rules"] = right_branch_children
                active_nodes.append(right_child_node)
            else:
                final_nodes.append(node)

        if not active_nodes:
            logger.debug("No expandable nodes remaining.")

        return root_node

    def _finalize(self):
        pass


def node_m_estimate(node: RuleNode) -> float:
    samples = node.samples
    parent_samples = node.parent.samples if node.parent else (0, 0)
    try:
        p = (parent_samples[0]) / sum(parent_samples)
        w = 1
        return math.log2((samples[0] + w * p) / (sum(samples) + w))
    except (ValueError, ZeroDivisionError):  # math domain error (from math.log2)
        return -9999


def cst_score(rule_stats: ChildRuleStats) -> Optional[float]:
    counts = rule_stats.counts
    try:
        # return (counts.ci + counts.cc + counts.ic, -(counts.cc), (counts.nmc - counts.ii))
        p = (counts.cc + counts.ic + counts.nmc) / counts.total
        w = 1
        return math.log2((counts.c_total + w * p) / (counts.m_total + w)) + (
            1 * math.log2(counts.m_total / counts.total)
        )  # This doesn't seem to add much.
    except (ValueError, ZeroDivisionError):  # math domain error (from math.log2)
        return None



