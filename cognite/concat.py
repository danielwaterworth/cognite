from cognite import expr
import mxnet as mx

def concat_fn(axis):
    def forward(args):
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
    return forward

def concat(args, axis):
    if all(lambda x: isinstance(expr.Constant), args):
        return expr.Constant(concat_fn(axis)(map(lambda x: x.value, args)))
    else:
        return expr.Apply(expr.Function('concat', concat_fn(axis)), [args])
