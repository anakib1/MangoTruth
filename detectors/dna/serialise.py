import pickle
from detectors.interfaces import SerializableConfig


def config_to_bytes(config: SerializableConfig) -> bytes:
    return pickle.dumps(config)


def config_from_bytes(config: bytes) -> SerializableConfig:
    return pickle.loads(config)
