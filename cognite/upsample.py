from cognite import expr
import mxnet as mx

def upsample_fn(scale):
    def forward(values):
        values = mx.ndarray.transpose(values, axes=(0, 3, 1, 2))
        output = \
            mx.ndarray.UpSampling(
                data = values,
                scale = scale,
                sample_type = 'nearest',
            )
        output = mx.ndarray.transpose(output, axes=(0, 2, 3, 1))
        def backwards(gradients):
            gradients = mx.ndarray.transpose(gradients, axes=(0, 3, 1, 2))
            output = \
                mx.ndarray.Pooling(
                    data = gradients,
                    kernel = (scale, scale),
                    pool_type = 'sum',
                    stride = (scale, scale),
                )
        return output, backwards
    return forward

def upsample(values, scale):
    if isinstance(values, expr.Constant):
        return expr.Constant(upsample_fn(scale)(values.value))
    else:
        return expr.Apply(expr.Function('upsample', upsample_fn(scale)), [values])
