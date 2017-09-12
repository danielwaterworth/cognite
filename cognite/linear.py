from cognite import expr
import mxnet as mx

class Linear(expr.Function):
    def forward(self, args):
        activations, weights = args
        output = mx.ndarray.dot(activations, weights)
        def backwards(gradients):
            activation_gradients = mx.ndarray.dot(gradients, weights.T)
            weight_gradients = mx.ndarray.dot(activations.T, gradients)
            return (activation_gradients, weight_gradients)
        return output, backwards

    def assert_output_shape(self, args, shape):
        activations, weights = args
        batch, output_size = shape

        try:
            act_shape = activations.get_shape()
        except expr.ShapeError:
            pass
        else:
            batch2, input_size = act_shape
            if batch != batch2:
                raise expr.ShapeError('Shape mismatch')
            weights.assert_shape((input_size, output_size))
            return None

        weight_shape = weights.get_shape()
        input_size, output_size2 = weight_shape
        if output_size2 != output_size:
            raise expr.ShapeError('Shape mismatch')
        activations.assert_shape((batch, input_size))

    def get_output_shape(self, args):
        activations, weights = args
        act_shape = activations.get_shape()
        if len(act_shape) != 2:
            raise expr.ShapeError('Cannot matrix multiply non-matrix')
        weight_shape = weights.get_shape()
        if len(weight_shape) != 2:
            raise expr.ShapeError('Cannot matrix multiply non-matrix')
        if act_shape[1] != weight_shape[0]:
            raise expr.ShapeError('Incompatible matrices to multiply')
        return (act_shape[0], weight_shape[1])

linear_fn = Linear()

def linear(x, weights):
    if isinstance(x, expr.Constant) and isinstance(weights, expr.Constant):
        return expr.Constant(linear_fn.forward([x.value, weights.value])[0])
    else:
        return expr.Apply(linear_fn, [x, weights])
