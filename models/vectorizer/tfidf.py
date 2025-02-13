from .vectorizer import Vectorizer

from typing import Iterable
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as sklearn_TfidfVectorizer

from ..utils import overrides, Serializable, Deserializable


class TfidfVectorizer(Vectorizer):

    _vectorizer: sklearn_TfidfVectorizer

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ):
        self._vectorizer = sklearn_TfidfVectorizer(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf)

    @overrides(Vectorizer)
    def fit(self, X: Iterable, y: Iterable | None = None) -> 'Vectorizer':
        self._vectorizer.fit(X)
        return self

    @overrides(Vectorizer)
    def transform(self, X: Iterable) -> Iterable:
        return self._vectorizer.transform(X)

    @overrides(Serializable)
    def serialize(self) -> bytes:
        return pickle.dumps(self._vectorizer)

    @overrides(Deserializable)
    def deserialize(self, data: bytes) -> None:
        self._vectorizer = pickle.loads(data)
