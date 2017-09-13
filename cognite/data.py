import functools
import mxnet as mx
import operator
import struct

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

def dump(data, fd):
    if isinstance(data, dict):
        fd.write(struct.pack('!B', 1))
        fd.write(struct.pack('!Q', len(data)))
        for key, value in data.items():
            assert isinstance(key, str)
            key_bytes = key.encode('utf8')
            fd.write(struct.pack('!Q', len(key_bytes)))
            fd.write(key_bytes)
            dump(value, fd)
    elif isinstance(data, mx.ndarray.NDArray):
        fd.write(struct.pack('!B', 2))
        fd.write(struct.pack('!Q', len(data.shape)))
        for x in data.shape:
            fd.write(struct.pack('!Q', x))
        size = functools.reduce(operator.mul, data.shape, 1)
        values = data.reshape((size,))
        for value in values.asnumpy():
            fd.write(struct.pack('!f', value))
    else:
        raise TypeError('not dumpable')

def load(fd):
    t = struct.unpack('!B', fd.read(1))[0]
    if t == 1:
        n = struct.unpack('!Q', fd.read(8))[0]
        output = {}
        for i in range(n):
            n = struct.unpack('!Q', fd.read(8))[0]
            key = fd.read(n).decode('utf8')
            value = load(fd)
            output[key] = value
        return output
    elif t == 2:
        n = struct.unpack('!Q', fd.read(8))[0]
        size = 1
        shape = []
        for i in range(n):
            n = struct.unpack('!Q', fd.read(8))[0]
            shape.append(n)
            size *= n
        values = []
        for i in range(size):
            f = struct.unpack('!f', fd.read(4))[0]
            values.append(f)
        return mx.ndarray.array(values).reshape(shape)
    else:
        raise TypeError('unknown data type')
