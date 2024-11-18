def ensure_type(obj, clazz):
    if not isinstance(obj, clazz):
        raise Exception(f"Object's type don't match. Expected {clazz}, but got {type(obj)}")


def ensure_obj(dct, handle):
    if handle not in dct:
        raise Exception(f"Dictionary is missing key {handle}.")

    return dct[handle]