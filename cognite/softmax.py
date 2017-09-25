from cognite import expr
import mxnet as mx

class Softmax(expr.Function):
    def forward(self, args):
        assert len(args) == 1
        x = args[0]

        output = mx.ndarray.softmax(x)
        def backward(gradients):
            raise NotImplementedError()
        return output, backward

    def assert_output_shape(self, args, shape):
        assert len(args) == 1
        x = args[0]

        x.assert_shape(shape)

    def get_output_shape(self, args):
        assert len(args) == 1
        x = args[0]

        return x.get_shape(x)

softmax_fn = Softmax()

def softmax(x):
    if isinstance(x, expr.Constant):
        return expr.Constant(softmax_fn.forward([x.value])[0])
    else:
        return expr.Apply(softmax_fn, [x])
