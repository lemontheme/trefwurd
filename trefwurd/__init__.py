import logging

from .defaults import LOGGING_LEVEL
from .lemmatizer import SpacyLemmatizer

logging.basicConfig(level=LOGGING_LEVEL)


