from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import Serializable, Deserializable, StrOverrider

from .metrics import Metrics

class Model(ABC, Serializable, Deserializable, StrOverrider):
    @abstractmethod
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None, test_size: float = None) -> Metrics | None:
        raise NotImplementedError

    @abstractmethod
    def inference(self, data: pd.DataFrame) -> Iterable:
        raise NotImplementedError

    def save(self, path: str | Path):
        with open(path, 'wb') as f:
            f.write(self.serialize())

    def load(self, path: str | Path):
        with open(path, 'rb') as f:
            self.deserialize(f.read())
