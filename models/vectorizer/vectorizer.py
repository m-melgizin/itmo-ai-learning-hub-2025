from abc import ABC, abstractmethod

from typing import Iterable

from ..utils import Serializable, Deserializable, StrOverrider

class Vectorizer(ABC, Serializable, Deserializable, StrOverrider):
    @abstractmethod
    def fit(self, X: Iterable, y: Iterable | None = None) -> 'Vectorizer':
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: Iterable) -> Iterable:
        raise NotImplementedError

    def fit_transform(self, X: Iterable, y: Iterable = None) -> Iterable:
        return self.fit(X, y).transform(X)
