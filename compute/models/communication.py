from uuid import uuid4
from dataclasses import dataclass
from typing import Mapping


@dataclass
class ComputeRequest:
    request_id: uuid4
    content: str
    detector_name: str


@dataclass
class ComputeResponse:
    explanation: str
    predictions: Mapping[str, float]
    request_id: str
