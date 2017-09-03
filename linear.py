import expr
import numpy as np

def forward(activations, weights):
    output = np.matmul(activations, weights)
    def backwards(gradients):
        activation_gradients = np.matmul(gradients, weights)
        weight_gradients = np.matmul(gradients, activations.T)
        return (activation_gradients, weight_gradients)
    return output, backwards

def linear(x, weights):
    if isinstance(x, expr.Constant) and isinstance(weights, expr.Constant):
        return expr.Constant(forward(x.value, weights.value)[0])
    else:
        return expr.Apply(expr.Function('linear', forward), [x, weights])
