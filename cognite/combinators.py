import mxnet as mx
from cognite import data

class Combinator:
    pass

class Identity(Combinator):
    def __init__(self, n):
        self.n = n

    @property
    def inputs(self):
        return self.n

    @property
    def outputs(self):
        return self.n

    def __call__(self, *scope):
        assert len(scope) == self.n
        def backwards(*gradients):
            assert len(gradients) == self.n
            return tuple(gradients)
        return tuple(scope), backwards

    def __repr__(self):
        return "Identity(%i)" % self.n

class Serial(Combinator):
    def __init__(self, *xs):
        assert len(xs) >= 1
        self.xs = xs

    @property
    def inputs(self):
        return self.xs[0].inputs

    @property
    def outputs(self):
        return self.xs[-1].outputs

    def __call__(self, *scope):
        backs = []
        intermediate = scope
        for fn in self.xs:
            intermediate, back = fn(*intermediate)
            backs.append(back)
        def backward(*gradient):
            intermediate = gradient
            for back in reversed(backs):
                intermediate = back(*intermediate)
            return intermediate
        return intermediate, backward

    def __repr__(self):
        return "Serial(%s)" % (', '.join(map(repr, self.xs)))

class Parallel(Combinator):
    def __init__(self, *xs):
        self.xs = xs

    @property
    def inputs(self):
        return sum([x.inputs for x in self.xs])

    @property
    def outputs(self):
        return sum([x.outputs for x in self.xs])

    def __call__(self, *scope):
        outputs = []
        backs = []
        for x in self.xs:
            output, back = x(*scope[:x.inputs])
            outputs.append(output)
            backs.append(back)
            scope = scope[x.inputs:]
        assert len(scope) == 0
        def backward(*gradients):
            outputs = []
            for back, x in zip(backs, self.xs):
                output = back(*gradients[:x.outputs])
                assert all(map(data.check, output)), repr(output)
                outputs.append(output)
                gradients = gradients[x.outputs:]
            assert len(scope) == 0
            result = tuple([x for xs in outputs for x in xs])
            return result
        return tuple([x for xs in outputs for x in xs]), backward

    def __repr__(self):
        return "Parallel(%s)" % (', '.join(map(repr, self.xs)))

class Discard(Combinator):
    def __init__(self, *mask):
        self.mask = mask

    @property
    def inputs(self):
        return 1

    @property
    def outputs(self):
        return 0

    def __call__(self, x):
        def backward():
            return data.zeros(x.shape)
        return (), backward

    def __repr__(self):
        return "Discard(%s)" % (', '.join(map(repr, self.mask)))

class Duplicate(Combinator):
    @property
    def inputs(self):
        return 1

    @property
    def outputs(self):
        return 2

    def __call__(self, x):
        def backward(a, b):
            return (data.add(a, b),)
        return (x, x), backward

    def __repr__(self):
        return "Duplicate()"

class Permutation(Combinator):
    def __init__(self, *indices):
        assert list(sorted(indices)) == list(range(len(indices)))
        self.indices = indices
        self.inverse = []
        for i in range(len(self.indices)):
            self.inverse.append(self.indices.index(i))

    @property
    def inputs(self):
        return len(self.indices)

    @property
    def outputs(self):
        return len(self.indices)

    def __call__(self, *scope):
        assert len(scope) == len(self.indices)
        output = []
        for index in self.indices:
            output.append(scope[index])
        def backward(*gradients):
            output = []
            for i in self.inverse:
                output.append(gradients[i])
            return tuple(output)
        return tuple(output), backward

    def __repr__(self):
        return "Permutation(%s)" % (', '.join(map(str, self.indices)))

class Apply(Combinator):
    def __init__(self, function, n):
        self.function = function
        self.n = n

    @property
    def inputs(self):
        return self.n

    @property
    def outputs(self):
        return 1

    def __call__(self, *scope):
        assert len(scope) == self.n
        output, back = self.function.forward(scope)
        return (output,), back

    def __repr__(self):
        return "Apply(%s, %d)" % (repr(self.function), self.n)

class Index(Combinator):
    def __init__(self, attr):
        self.attr = attr

    @property
    def inputs(self):
        return 1

    @property
    def outputs(self):
        return 1

    def __call__(self, value):
        def back(gradients):
            return ({self.attr: gradients},)
        return (value[self.attr],), back

    def __repr__(self):
        return "Index(%s)" % self.attr

class Constant(Combinator):
    def __init__(self, value):
        self.value = value

    @property
    def inputs(self):
        return 0

    @property
    def outputs(self):
        return 1

    def __call__(self):
        def back(gradients):
            return ()
        return (self.value,), back

    def __repr__(self):
        return "Constant()"
