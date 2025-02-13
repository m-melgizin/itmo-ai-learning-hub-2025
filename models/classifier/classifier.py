from abc import ABC, abstractmethod

from typing import Iterable

from ..utils import Serializable, Deserializable, StrOverrider

class Classifier(ABC, Serializable, Deserializable, StrOverrider):
    @abstractmethod
    def fit(self, X: Iterable, y: Iterable) -> 'Classifier':
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X: Iterable) -> Iterable:
        raise NotImplementedError
