from typing import Dict, Any, Union
from pathlib import Path
import importlib.resources

from .lemmatizer import Lemmatizer


data = importlib.resources.contents("trefwurd.data.xx")

breakpoint()


def load(path_or_lang: Union[Path, str]) -> lemmatizer.Lemmatizer:
    pass


def info(lang: str) -> Dict[str, Any]:
    pass


def set_log_level():
    pass
