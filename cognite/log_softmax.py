from cognite import expr
import mxnet as mx

class SoftmaxCrossEntropy(expr.Function):
    def forward(self, args):
        x, labels = args

        def backward(gradients):
            t = mx.ndarray.softmax(x)
            return mx.ndarray.multiply(gradients, mx.ndarray.subtract(t, labels))

        output = mx.ndarray.softmax_cross_entropy(x, labels)
        return output, backward

    def assert_output_shape(self, args, shape):
        a, b = args
        a.assert_shape(shape)
        b.assert_shape(shape)

    def get_output_shape(self, args):
        a, b = args
        try:
            shape = a.get_shape()
        except expr.ShapeError:
            pass
        else:
            b.assert_shape(shape)
            return shape

        shape = b.get_shape()
        a.assert_shape(shape)
        return shape

softmax_cross_entropy_fn = SoftmaxCrossEntropy()

def softmax_cross_entropy(x, labels):
    if isinstance(x, expr.Constant) and isinstance(labels, expr.Constant):
        return expr.Constant(softmax_cross_entropy_fn.forward([x.value, labels.value])[0])
    else:
        return expr.Apply(softmax_cross_entropy_fn, [x, labels])
