from cognite import expr
import mxnet as mx
import operator

def num_values(shape):
    return reduce(operator.mul, values.get_shape(), 1)

class Reshape(expr.Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, args):
        assert len(args) == 1
        values = args[0]

        output = mx.ndarray.reshape(values, self.shape)
        def backwards(gradients):
            return mx.ndarray.reshape(gradients, values.shape)
        return output, backwards

    def get_output_shape(self, args):
        assert len(args) == 1
        values = args[0]
        input_shape = values.get_shape()
        if num_values(shape) != num_values(input_shape):
            raise expr.ShapeError('Incompatible shapes for reshaping %s and %s' % (input_shape, shape))
        return self.shape

def reshape(values, new_shape):
    if isinstance(values, expr.Constant):
        return expr.Constant(Reshape(new_shape).forward([values.value]))
    else:
        return expr.Apply(Reshape(new_shape), [values])
