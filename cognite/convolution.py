from cognite import expr
import mxnet as mx

def convolution(weight, activations):
    height, width, channels_out, channels_in = weight.shape
    weight = mx.ndarray.transpose(weight, axes=(2, 3, 0, 1))
    activations = mx.ndarray.transpose(activations, axes=(0, 3, 1, 2))
    output = \
        mx.ndarray.Convolution(
            kernel = (height, width),
            num_filter = channels_out,
            data = activations,
            weight = weight,
            no_bias = True,
        )
    return mx.ndarray.transpose(output, axes=(0, 2, 3, 1))

def convolution_input_gradient(delta, weight):
    height, width, channels_out, channels_in = weight.shape
    delta = mx.ndarray.transpose(output, axes=(0, 3, 1, 2))
    weight = mx.ndarray.transpose(weight, axes=(2, 3, 0, 1))
    input_gradient = \
        mx.ndarray.Deconvolution(
            kernel = (height, width),
            num_filter = channels_in,
            data = delta,
            weight = weight,
            no_bias = True,
        )
    return mx.ndarray.transpose(input_gradient, axes=(0, 2, 3, 1))

def convolution_weight_gradient(delta, activations):
    batch_size, height, width, channels = activations.shape
    delta = mx.ndarray.transpose(delta, axes=(3, 0, 1, 2))
    activations = mx.ndarray.transpose(activations, axes=(0, 3, 1, 2))
    weight_gradient = \
        mx.ndarray.Deconvolution(
            kernel = (height, width),
            num_filter = channels,
            data = delta,
            weight = activations,
            no_bias = True,
        )
    return mx.ndarray.transpose(weight_gradient, axes=(2, 3, 0, 1))

class Convolution(expr.Function):
    def __init__(self, kernel, outputs):
        self.kernel = kernel
        self.outputs = outputs

    def forward(self, args):
        activations, weights = args
        output = convolution(activations, weights)
        def backwards(gradients):
            activation_gradients = convolution_input_gradient(gradients, weights)
            weight_gradients = convolution_weight_gradient(gradients, activations)
            return (activation_gradients, weight_gradients)
        return output, backwards

    def get_output_shape(self, args):
        activations, weights = args
        act_shape = activations.get_shape()
        if len(act_shape) != 4:
            raise expr.ShapeError('expected activations to be 4D, DHWC')
        batch, height, width, in_channels = activations.shape
        filter_height, filter_width = self.kernel
        weights.assert_shape((filter_height, filter_width, self.outputs, in_channels))
        return (batch, height - filter_height + 1, width - filter_width + 1, self.outputs)

def convolution2d(x, weights):
    assert False
    if isinstance(x, expr.Constant) and isinstance(weights, expr.Constant):
        return expr.Constant(forward(x.value, weights.value)[0])
    else:
        return expr.Apply(expr.Function('convolution2d', forward), [x, weights])
