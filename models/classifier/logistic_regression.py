from .classifier import Classifier

import pickle
from sklearn.linear_model import LogisticRegression as sklearn_LogisticRegression

from typing import Iterable

from ..utils import overrides, Serializable, Deserializable

class LogisticRegression(Classifier):

    _classifier: sklearn_LogisticRegression

    def __init__(
        self,
        penalty="l2",
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        self._classifier = sklearn_LogisticRegression(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio
        )
    
    @overrides(Classifier)
    def fit(self, X: Iterable, y: Iterable) -> 'Classifier':
        self._classifier.fit(X, y)
        return self

    @overrides(Classifier)
    def predict(self, X: Iterable) -> Iterable:
        return self._classifier.predict(X)
    
    @overrides(Serializable)
    def serialize(self) -> bytes:
        return pickle.dumps(self._classifier)
    
    @overrides(Deserializable)
    def deserialize(self, data: bytes) -> None:
        self._classifier = pickle.loads(data)
