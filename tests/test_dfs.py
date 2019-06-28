from trefwurd.dfs import dfs_graph_best_path


def test_dfs_path():
    simple_graph = {
        "a": ["b"],
        "b": ["c", "d", "e"],
        "c": ["p", "g", "k"],
        "k": ["l"],
        "e": [10]
    }

    path = dfs_graph_best_path(simple_graph, "a", lambda v: v != 10)
    assert path == [("a", True), ("b", True), ("e", True)]
