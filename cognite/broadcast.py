from cognite import expr
import mxnet as mx

def broadcasted_dims(fr, to):
    assert len(to) >= len(fr)
    fr = [1]*(len(to) - len(fr)) + list(fr)
    dims = []
    for i, (x, y) in enumerate(zip(fr, to)):
        if x == 1:
            if y != 1:
                dims.append(i)
        else:
            assert x == y
    return dims

class Broadcast(expr.Function):
    def __init__(self, shape=None):
        self.shape = shape

    def forward(self, args):
        assert len(args) == 1
        x = args[0]

        output = x.broadcast_to(self.shape)
        def backward(gradient):
            dims = broadcasted_dims(x.shape, self.shape)
            values = mx.ndarray.sum(gradient, axis=dims, keepdims=True)
            return (mx.ndarray.reshape(values, x.shape),)
        return output, backward

    def assert_output_shape(self, args, shape):
        assert len(args) == 1
        x = args[0]

        if self.shape is not None:
            if self.shape != shape:
                raise expr.ShapeError('mismatched shape %s and %s' % (self.shape, shape))

        broadcasted_dims(x.get_shape(), shape)

    def get_output_shape(self, args):
        assert len(args) == 1
        x = args[0]

        if self.shape is None:
            raise expr.ShapeError('Unknown shape')

        broadcasted_dims(x.get_shape(), self.shape)
        return self.shape

def broadcast(x, shape=None):
    if isinstance(x, expr.Constant):
        return expr.Constant(Broadcast(shape).forward([x.value])[0])
    else:
        return expr.Apply(Broadcast(shape), [x])
