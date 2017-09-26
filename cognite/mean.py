from cognite import expr
import mxnet as mx

class Mean(expr.Function):
    def __init__(self, dims):
        assert isinstance(dims, tuple)
        self.dims = dims

    def forward(self, args):
        assert len(args) == 1
        values = args[0]

        mult = 1
        for dim in self.dims:
            mult *= values.shape[dim]
        def backwards(gradients):
            return (mx.ndarray.broadcast_to(mx.ndarray.divide(gradients, mult), values.shape),)
        output = mx.ndarray.mean(values, self.dims, keepdims=True)
        return output, backwards

    def get_output_shape(self, args):
        assert len(args) == 1
        values = args[0]

        shape = list(values.get_shape())
        for dim in self.dims:
            shape[dim] = 1
        return shape

def mean(values, dims):
    if isinstance(values, expr.Constant):
        return expr.Constant(Mean(dims).forward(values.value))
    else:
        return expr.Apply(Mean(dims), [values])
