from dataclasses import dataclass

@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
