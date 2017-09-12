import mxnet as mx

def elementwise(a, b, fn):
    assert (isinstance(a, dict) and isinstance(b, dict)) or (isinstance(a, mx.ndarray.NDArray) and isinstance(b, mx.ndarray.NDArray))
    if isinstance(a, dict):
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        output = {}
        for key in a_keys - b_keys:
            output[key] = a[key]
        for key in b_keys - a_keys:
            output[key] = b[key]
        for key in a_keys & b_keys:
            output[key] = elementwise(a[key], b[key], fn)
        return output
    else:
        return fn(a, b)

def add(a, b):
    return elementwise(a, b, mx.ndarray.add)

def subtract(a, b):
    return elementwise(a, b, mx.ndarray.subtract)

def multiply(a, b):
    return elementwise(a, b, mx.ndarray.multiply)

def zeros(shape):
    if isinstance(shape, dict):
        output = {}
        for key, value in shape.items():
            output[key] = zeros(value)
        return output
    else:
        return mx.ndarray.zeros(shape)

def check(data):
    if isinstance(data, mx.ndarray.NDArray):
        return True
    elif isinstance(data, dict):
        return \
            all(map(lambda x: isinstance(x, str), data.keys())) and \
            all(map(check, data.values()))
    else:
        return False
