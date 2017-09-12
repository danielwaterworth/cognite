from cognite import expr
import mxnet as mx

class Relu(expr.Function):
    def forward(self, args):
        assert len(args) == 1
        x = args[0]

        mask = x > 0
        output = mx.ndarray.multiply(mask, x)
        def backward(gradient):
            return (mx.ndarray.multiply(gradient, mask),)
        return output, backward

    def assert_output_shape(self, args, shape):
        assert len(args) == 1
        x = args[0]

        return x.assert_shape(shape)

    def get_output_shape(self, args):
        assert len(args) == 1
        x = args[0]

        return x.get_shape()

relu_fn = Relu()

def relu(x):
    if isinstance(x, expr.Constant):
        return expr.Constant(relu_fn.forward([x.value])[0])
    else:
        return expr.Apply(relu_fn, [x])
