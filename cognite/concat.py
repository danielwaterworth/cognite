from cognite import expr
import mxnet as mx

class Concat(expr.Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, args):
        output = mx.ndarray.concat(args, dim=axis)
        def backwards(gradients):
            outputs = []
            n = 0
            for arg in args:
                size = arg.shape[axis]
                begin = n
                end = begin+size
                n = end
                outputs.append(
                    mx.ndarray.slice_axis(
                        gradients,
                        axis=axis,
                        begin=begin,
                        end=end,
                    )
                )
            return outputs
        return output, backwards

    def get_output_shape(self, args):
        raise NotImplementedError()

def concat(args, axis):
    if all(lambda x: isinstance(expr.Constant), args):
        return expr.Constant(Concat(axis).forward(map(lambda x: x.value, args)))
    else:
        return expr.Apply(Concat(axis), [args])
