import meta
import expr
import numpy as np

@meta.differentiable_function
def linear_layer(x, weights, biases):
    x = meta.linear(x, weights)
    x = meta.add(x, biases)
    return meta.relu(x)

output, backward = linear_layer(np.array([[1], [2]]), np.array([[2]]), np.array([3]))
print(output[0])
print(backward(np.array([[1], [1]])))
