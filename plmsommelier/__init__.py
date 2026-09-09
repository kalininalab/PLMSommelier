"""plmsommelier -- pick the best layer of a protein language model for your dataset.

Implements the practical consequence of Joeres, Senatorov et al.,
*Task- and dataset-specific information in protein language models*
(arXiv:2608.12090): the last layer is almost never the best one, so probe a
subsample of your data and ship a truncated encoder instead.
"""

from __future__ import annotations

__version__ = "1.0.0"

from plmsommelier.data import Dataset, load_dataset
from plmsommelier.model import PLM, embed_layers, load_model, save_truncated, truncate
from plmsommelier.select import Result, select_layer

__all__ = [
    "__version__",
    "Dataset",
    "load_dataset",
    "PLM",
    "load_model",
    "embed_layers",
    "truncate",
    "save_truncated",
    "Result",
    "select_layer",
]
