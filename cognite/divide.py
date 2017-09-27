from cognite import expr
import mxnet as mx

class Divide(expr.Function):
    def forward(self, args):
        a, b = args

        b_recip = mx.ndarray.reciprocal(b)
        output = mx.ndarray.multiply(a, b_recip)
        def backward(gradients):
            b_gradient = mx.ndarray.multiply(output, b_recip)
            b_gradient = mx.ndarray.multiply(-1, b_gradient)
            return (b_recip, b_gradient)
        return output, backward

    def assert_output_shape(self, args, shape):
        a, b = args
        a.assert_shape(shape)
        b.assert_shape(shape)

    def get_output_shape(self, args):
        a, b = args
        try:
            shape = a.get_shape()
        except expr.ShapeError:
            pass
        else:
            b.assert_shape(shape)
            return shape

        shape = b.get_shape()
        a.assert_shape(shape)
        return shape

divide_fn = Divide()

def divide(x, biases):
    if isinstance(x, expr.Constant) and isinstance(biases, expr.Constant):
        return expr.Constant(divide_fn.forward([x.value, biases.value])[0])
    else:
        return expr.Apply(divide_fn, [x, biases])
