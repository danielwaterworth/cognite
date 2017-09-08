from cognite import expr
import mxnet as mx

def forward(x):
    mask = x > 0
    output = mx.ndarray.multiply(mask, x)
    def backward(gradient):
        return (mx.ndarray.multiply(gradient, mask),)
    return output, backward

def relu(x):
    if isinstance(x, expr.Constant):
        return expr.Constant(forward(x.value)[0])
    else:
        return expr.Apply(expr.Function('relu', forward), [x])
