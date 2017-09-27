from cognite import expr
import mxnet as mx

class Sqrt(expr.Function):
    def forward(self, args):
        assert len(args) == 1
        x = args[0]

        s = mx.ndarray.sqrt(x)
        def backward(gradient):
            output = \
                mx.ndarray.multiply(
                    mx.ndarray.multiply(
                        mx.ndarray.power(x, -0.5),
                        0.5,
                    ),
                    gradient,
                )
            return (output,)
        return s, backward

    def assert_output_shape(self, args, shape):
        assert len(args) == 1
        x = args[0]

        return x.assert_shape(shape)

    def get_output_shape(self, args):
        assert len(args) == 1
        x = args[0]

        return x.get_shape()

sqrt_fn = Sqrt()

def sqrt(x):
    if isinstance(x, expr.Constant):
        return expr.Constant(sqrt_fn.forward([x.value])[0])
    else:
        return expr.Apply(sqrt_fn, [x])
