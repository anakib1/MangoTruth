import json
from dataclasses import dataclass
from uuid import uuid4


@dataclass
class ComputeRequest:
    request_id: uuid4
    content: str
    detector_name: str

    @classmethod
    def from_json(cls, json_data):
        # Parse if json_data is a string; assume it's already a dict if not
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        return cls(
            request_id=data['request_id'],
            content=data['content'],
            detector_name=data.get('detector_name', None)
        )


@dataclass
class ComputeResponse:
    request_id: str
    status: str
    verdict: dict[str, list[dict[str, float]]]
