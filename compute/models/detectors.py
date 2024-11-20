from dataclasses import dataclass
from uuid import uuid4


@dataclass
class Detector:
    run_id: uuid4
    name: str
    classpath: str
