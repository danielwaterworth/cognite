from cognite import expr
import mxnet as mx

class Subtract(expr.Function):
    def forward(self, args):
        activations, biases = args

        output = mx.ndarray.subtract(activations, biases)
        def backward(gradients):
            return (gradients, -gradients)
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

subtract_fn = Subtract()

def subtract(x, biases):
    if isinstance(x, expr.Constant) and isinstance(biases, expr.Constant):
        return expr.Constant(subtract_fn.forward([x.value, biases.value])[0])
    else:
        return expr.Apply(subtract_fn, [x, biases])
