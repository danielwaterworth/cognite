from cognite import expr
import mxnet as mx

def forward(activations, weights):
    output = mx.ndarray.dot(activations, weights)
    def backwards(gradients):
        activation_gradients = mx.ndarray.dot(gradients, weights.T)
        weight_gradients = mx.ndarray.dot(activations.T, gradients)
        return (activation_gradients, weight_gradients)
    return output, backwards

def linear(x, weights):
    if isinstance(x, expr.Constant) and isinstance(weights, expr.Constant):
        return expr.Constant(forward(x.value, weights.value)[0])
    else:
        return expr.Apply(expr.Function('linear', forward), [x, weights])
