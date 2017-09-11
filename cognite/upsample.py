from cognite import expr
import mxnet as mx

class Upsample(expr.Function):
    def __init__(self, scale):
        self.scale = scale

    def forward(self, args):
        assert len(args) == 1
        values = args[0]

        values = mx.ndarray.transpose(values, axes=(0, 3, 1, 2))
        output = \
            mx.ndarray.UpSampling(
                data = values,
                scale = self.scale,
                sample_type = 'nearest',
            )
        output = mx.ndarray.transpose(output, axes=(0, 2, 3, 1))
        def backwards(gradients):
            gradients = mx.ndarray.transpose(gradients, axes=(0, 3, 1, 2))
            output = \
                mx.ndarray.Pooling(
                    data = gradients,
                    kernel = (self.scale, self.scale),
                    pool_type = 'sum',
                    stride = (self.scale, self.scale),
                )
        return output, backwards

def upsample(values, scale):
    if isinstance(values, expr.Constant):
        return expr.Constant(Upsample(scale).forward(values.value))
    else:
        return expr.Apply(Upsample(scale), [values])
