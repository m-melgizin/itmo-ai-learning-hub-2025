from .model import Model

from typing import Iterable

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd

from .utils import overrides, Serializable, Deserializable, StrOverrider

from .metrics import Metrics
from .vectorizer import Vectorizer
from .classifier import Classifier


class VectorizerWithClassifier(Model):

    _vectorizer: Vectorizer
    _classifier: Classifier

    def __init__(self, vectorizer: Vectorizer, classifier: Classifier):
        self._vectorizer = vectorizer
        self._classifier = classifier

    @overrides(Model)
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None, test_size: float = None) -> Metrics | None:
        X = train_data['text']
        y = train_data['label']

        if val_data is not None:
            X_train = X
            y_train = y
            X_val = val_data['text']
            y_val = val_data['label']
        elif test_size is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=17)
        else:
            X_train = X
            y_train = y

        X_train = self._vectorizer.fit_transform(X_train)
        X_val = self._vectorizer.transform(
            X_val) if X_val is not None else None

        self._classifier.fit(X_train, y_train)

        if X_val is None:
            return None

        y_pred = self._classifier.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='macro')
        recall = recall_score(y_val, y_pred, average='macro')
        f1 = f1_score(y_val, y_pred, average='macro')

        return Metrics(accuracy, precision, recall, f1)

    @overrides(Model)
    def inference(self, data: pd.DataFrame) -> Iterable:
        X = data['text']
        X = self._vectorizer.transform(X)
        y_pred = self._classifier.predict(X)
        return y_pred

    @overrides(Serializable)
    def serialize(self) -> bytes:
        vectorizer_bytes = self._vectorizer.serialize()
        vectorizer_bytes_size = len(vectorizer_bytes).to_bytes(8, 'big')
        model_bytes = self._classifier.serialize()
        model_bytes_size = len(model_bytes).to_bytes(8, 'big')

        return vectorizer_bytes_size + vectorizer_bytes + model_bytes_size + model_bytes

    @overrides(Deserializable)
    def deserialize(self, data: bytes) -> None:
        vectorizer_bytes_size = int.from_bytes(data[:8], 'big')
        vectorizer_bytes = data[8:8+vectorizer_bytes_size]
        self._vectorizer.deserialize(vectorizer_bytes)

        model_bytes_size = int.from_bytes(
            data[8+vectorizer_bytes_size:8+vectorizer_bytes_size+8], 'big')
        model_bytes = data[8+vectorizer_bytes_size +
                           8:8+vectorizer_bytes_size+8+model_bytes_size]
        self._vectorizer.deserialize(model_bytes)

    @overrides(StrOverrider)
    def __str__(self) -> str:
        return 'VectorizerWithModel(vectorizer={}, classifier={})'.format(str(self._vectorizer), str(self._classifier))
