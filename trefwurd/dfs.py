"""Depth-first search"""

from typing import Dict, Any, List, Hashable, Callable, Tuple


def dfs_graph_best_path(graph: Dict[Hashable, List[Hashable]],
                        start_node: Hashable,
                        predicate: Callable[[Any], Any]) -> List[Tuple[Any, Any]]:
    stack = [(start_node, [(start_node, None)])]
    visited = set()
    best_path = None
    while stack:
        (node, path) = stack.pop()
        if node in visited:
            continue
        v = predicate(node)
        path[-1] = (node, v)
        if v:
            best_path = path
        elif best_path and len(path) > len(best_path):
            break
        visited.add(node)
        neighbors = graph.get(node, [])
        for neighbor in neighbors:
            stack.append((neighbor, path + [(neighbor, None)]))
    return best_path


