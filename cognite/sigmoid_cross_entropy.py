from cognite import expr
import mxnet as mx

class SigmoidCrossEntropy(expr.Function):
    def forward(self, args):
        x, labels = args
        assert(x.shape == labels.shape)

        def backward(gradients):
            t = mx.ndarray.sigmoid(x)
            x_gradient = mx.ndarray.multiply(gradients, mx.ndarray.subtract(t, labels))

            # FIXME:
            label_gradient = mx.ndarray.zeros(labels.shape)
            return (x_gradient, label_gradient)

        output = \
            mx.ndarray.LogisticRegressionOutput(
                data=x,
                labels=labels,
            )

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

sigmoid_cross_entropy_fn = SigmoidCrossEntropy()

def sigmoid_cross_entropy(x, labels):
    if isinstance(x, expr.Constant) and isinstance(labels, expr.Constant):
        return expr.Constant(sigmoid_cross_entropy_fn.forward([x.value, labels.value])[0])
    else:
        return expr.Apply(sigmoid_cross_entropy_fn, [x, labels])
