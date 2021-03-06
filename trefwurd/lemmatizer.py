import collections.abc
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Sequence, Optional, Dict, Union, Any, Iterable, Hashable, Type
import warnings

from .tree import RuleTree, RuleTreeForest
from .utils import create_file, create_dir, generate_meta

import srsly


Model = Union[RuleTree, RuleTreeForest]


class Lemmatizer:

    def __init__(self, lang: str = None, meta=None, **cfg):
        self.meta = meta if meta is not None else {}
        if lang and "lang" not in self.meta:
            self.meta["lang"] = lang
        self.cfg = cfg
        # -------------------------------------------
        self._feat2model: Dict[Union[Hashable, None], Model] = {}
        self._feat2lookup = {}
        self._lemmatize = None
        self._model_cls: Type[Model] = RuleTreeForest
        self._exceptions = None

    @property
    def lang(self):
        return self.meta.get("lang")

    def lemmatize(self, s, pos=None, morph=None):
        try:
            return self._lemmatize(s)
        except TypeError:
            raise NotImplementedError("Concrete implementation set by calling fit() first.")

    def fit(
        self, x: Iterable[Union[str, Tuple[str, str], Tuple[str, str, str]]], y: Iterable[str]
    ) -> "Lemmatizer":
        """
        Args:
            x: tokens, (pos), (morph)
            y: lemmas
        Returns: Lemmatizer
        """
        if not isinstance(x, collections.abc.Sequence):
            x = list(x)
        if not isinstance(y, collections.abc.Sequence):
            y = list(y)
        if not len(x) == len(y):
            raise ValueError("`x` and `y` args should have the same length.")
        self._feat2lookup = {}
        n_features = len(x[0][1:])
        if n_features == 0:
            self._fit_no_pos(x, y)  # Raises NotImplementedError.
        elif n_features == 1:
            self._fit_w_pos(x, y)
        elif n_features == 2:
            self._fit_w_morph(x, y)  # Raises NotImplementedError.
        else:
            raise ValueError("Invalid `x` argument")
        return self

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(exist_ok=True)
        create_file(path / "__init__.py")
        meta = generate_meta(self.meta)
        srsly.write_json(path / "meta.json", meta)
        model_dir = path / "data"
        model_dir.mkdir(exist_ok=True)
        srsly.write_json(model_dir / "cfg", self.cfg)
        srsly.write_gzip_json(model_dir / "lookup.gz", self._feat2lookup or {})
        srsly.write_gzip_json(model_dir / "model.gz", self._feat2model)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Lemmatizer":
        path = Path(path)
        if not path.is_dir():
            raise ValueError(f"{path} must be a directory or archive")
        model_dir = path / "data"
        meta = path / "meta.json"
        meta = srsly.read_json(meta)
        cfg = srsly.read_json(model_dir / "cfg")
        lookup = srsly.read_gzip_json(model_dir / "lookup.gz")
        model = srsly.read_gzip_json(model_dir / "model.gz")
        lemmatizer = cls()
        lemmatizer._meta = meta
        lemmatizer.cfg = cfg
        lemmatizer._feat2model = model
        lemmatizer._feat2lookup = lookup
        return lemmatizer

    def _to_dict(self):
        return {
            "cfg": self.cfg,
            "meta": self.meta,
            "rules": self._feat2model,
            "lookup": self._feat2lookup,
        }

    def _fit_no_pos(self, x: Sequence[str], y: Sequence[str]):
        model = self._model_cls(**self.cfg)
        model.fit(x, y)
        self._feat2model[None] = model
        self.lemmatize = self._lemmatize_no_pos
        return self

    def _fit_w_pos(self, x: Sequence[Tuple[str, str]], y: Sequence[str]) -> "Lemmatizer":
        examples_by_pos = defaultdict(dict)  # type: Dict[str, Dict[str, str]]
        for (token, pos), lemma in zip(x, y):
            examples_by_pos[pos][token] = lemma
        for pos, examples in examples_by_pos.items():
            model = self._model_cls(**self.cfg)  # type: Union[RuleTree, RuleTreeForest]
            x_pos, y_pos = zip(*examples.items())
            model.fit(x_pos, y_pos)
            self._feat2model[pos] = model.to_dict()
        self.lemmatize = self._lemmatize_w_pos
        return self

    def _fit_w_morph(self, x: Sequence[Tuple[str, str, str]], y: Sequence[str]) -> "Lemmatizer":
        # return self
        raise NotImplementedError

    def _optimize(self):
        pass

    def _lemmatize_no_pos(self, s):
        return self._feat2model[None].predict(s)

    def _lemmatize_w_pos(self, s, pos):
        try:
            return self._feat2model[pos].predict(s)
        except KeyError:
            warnings.warn(f"No model found for pos/feature tag ({pos}).")
            return s

    def _lemmatize_w_morph(self, pos, morph=None):
        raise NotImplementedError

