from cognite import expr
import numpy as np

def forward(x):
    assert isinstance(x, np.ndarray)

    mask = x > 0
    output = mask * x
    def backward(gradient):
        return (gradient * mask,)
    return output, backward

def relu(x):
    if isinstance(x, expr.Constant):
        return expr.Constant(forward(x.value)[0])
    else:
        return expr.Apply(expr.Function('relu', forward), [x])
