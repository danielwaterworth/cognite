import expr
import numpy as np

def forward(activations, biases):
    output = activations + biases
    def backward(gradients):
        return (gradients, gradients)
    return output, backward

def add(x, biases):
    if isinstance(x, expr.Constant) and isinstance(biases, expr.Constant):
        return expr.Constant(forward(x.value, biases.value)[0])
    else:
        return expr.Apply(expr.Function('add', forward), [x, biases])
