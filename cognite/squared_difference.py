from cognite import expr
import numpy as np

def forward(a, b):
    difference = a - b
    output = difference**2
    def backwards(gradients):
        return (2*difference, -2*difference)
    return output, backwards

def squared_difference(x, weights):
    if isinstance(x, expr.Constant) and isinstance(weights, expr.Constant):
        return expr.Constant(forward(x.value, weights.value)[0])
    else:
        return expr.Apply(expr.Function('squared_difference', forward), [x, weights])
