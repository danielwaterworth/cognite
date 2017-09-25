from cognite import expr
import mxnet as mx

class AddBiases(expr.Function):
    def forward(self, args):
        activations, biases = args

        output = mx.ndarray.broadcast_add(activations, biases)
        def backward(gradients):
            summed_gradients = \
                mx.ndarray.sum(gradients, axis=tuple(range(len(gradients.shape)-1)))
            return (gradients, summed_gradients)
        return output, backward

    def assert_output_shape(self, args, shape):
        a, b = args
        a.assert_shape(shape)
        biases_shape = (shape[-1],)
        b.assert_shape(biases_shape)
        b.set_initializer(lambda : mx.nd.zeros(biases_shape))

    def get_output_shape(self, args):
        a, b = args
        shape = a.get_shape()
        biases_shape = (shape[-1],)
        b.assert_shape(biases_shape)
        b.set_initializer(lambda : mx.nd.zeros(biases_shape))
        return shape

add_biases_fn = AddBiases()

def add_biases(x, biases):
    if isinstance(x, expr.Constant) and isinstance(biases, expr.Constant):
        return expr.Constant(add_biases_fn.forward([x.value, biases.value])[0])
    else:
        return expr.Apply(add_biases_fn, [x, biases])
