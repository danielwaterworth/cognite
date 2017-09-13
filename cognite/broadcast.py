from cognite import expr
import mxnet as mx

def broadcasted_dims(fr, to):
    assert len(fr) == len(to)
    dims = []
    for i, (x, y) in enumerate(zip(fr, to)):
        if x == 1:
            if y != 1:
                dims.append(i)
        else:
            assert x == y
    return dims

class Broadcast(expr.Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, args):
        assert len(args) == 1
        x = args[0]

        output = x.broadcast_to(self.shape)
        def backward(gradient):
            dims = broadcasted_dims(x.shape, self.shape)
            return (mx.ndarray.sum(gradient, axis=dims, keepdims=True),)
        return output, backward

    def get_output_shape(self, args):
        assert len(args) == 1
        x = args[0]

        broadcasted_dims(x.get_shape(), self.shape)
        return self.shape

def broadcast(x, shape):
    if isinstance(x, expr.Constant):
        return expr.Constant(Broadcast(shape).forward([x.value])[0])
    else:
        return expr.Apply(Broadcast(shape), [x])
