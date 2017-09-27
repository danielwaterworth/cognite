from cognite import expr
import mxnet as mx

class Sigmoid(expr.Function):
    def forward(self, args):
        assert len(args) == 1
        x = args[0]

        s = mx.ndarray.sigmoid(x)
        def backward(gradient):
            return (s*(1-s),)
        return s, backward

    def assert_output_shape(self, args, shape):
        assert len(args) == 1
        x = args[0]

        return x.assert_shape(shape)

    def get_output_shape(self, args):
        assert len(args) == 1
        x = args[0]

        return x.get_shape()

sigmoid_fn = Sigmoid()

def sigmoid(x):
    if isinstance(x, expr.Constant):
        return expr.Constant(sigmoid_fn.forward([x.value])[0])
    else:
        return expr.Apply(sigmoid_fn, [x])
