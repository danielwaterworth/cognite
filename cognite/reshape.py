from cognite import expr
import mxnet as mx

def reshape_fn(new_shape):
    def forward(values):
        output = mx.ndarray.reshape(values, new_shape)
        def backwards(gradients):
            return mx.ndarray.reshape(gradients, values.shape)
        return output, backwards
    return forward

def reshape(values, new_shape):
    if isinstance(values, expr.Constant):
        return expr.Constant(reshape_fn(new_shape)(values.value))
    else:
        return expr.Apply(expr.Function('reshape', reshape_fn(new_shape)), [values])
