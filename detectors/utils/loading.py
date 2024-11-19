from typing import Any, Dict


def ensure_type(obj: Any, clazz) -> Any:
    if not isinstance(obj, clazz):
        raise Exception(f"Object's type don't match. Expected {clazz}, but got {type(obj)}")
    return obj


def ensure_obj(dct: Dict, handle: str) -> Any:
    if handle not in dct:
        raise Exception(f"Dictionary is missing key {handle}.")

    return dct[handle]
