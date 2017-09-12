from cognite import expr
import numpy as np

class SquaredDifference(expr.Function):
    def forward(self, args):
        a, b = args
        difference = a - b
        output = difference**2
        def backwards(gradients):
            return (2*difference, -2*difference)
        return output, backwards

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

squared_difference_fn = SquaredDifference()

def squared_difference(x, weights):
    if isinstance(x, expr.Constant) and isinstance(weights, expr.Constant):
        return expr.Constant(squared_difference_fn.forward([x.value, weights.value])[0])
    else:
        return expr.Apply(squared_difference_fn, [x, weights])
