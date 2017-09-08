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
        return len(self.mask)

    @property
    def outputs(self):
        return len(filter(lambda x: not x, self.mask))

    def __call__(self, *scope):
        assert len(scope) == len(self.mask)
        output = []
        for m, x in zip(self.mask, scope):
            if not m:
                output.append(x)
        def backward(*gradients):
            raise NotImplementedError()
        return tuple(output), backward

    def __repr__(self):
        return "Discard(%s)" % (', '.join(map(repr, self.mask)))

class Duplicate(Combinator):
    def __init__(self, *mask):
        self.mask = mask

    @property
    def inputs(self):
        return len(self.mask)

    @property
    def outputs(self):
        return len(self.mask) + len(filter(lambda x: x, self.mask))

    def __call__(self, *scope):
        assert len(scope) == len(self.mask)
        output = []
        for m, x in zip(self.mask, scope):
            output.append(x)
            if m:
                output.append(x)
        def backward(*gradients):
            raise NotImplementedError()
        return tuple(output), backward

    def __repr__(self):
        return "Duplicate(%s)" % (', '.join(map(repr, self.mask)))

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
            return output
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
        output, back = self.function(*scope)
        return (output,), back

    def __repr__(self):
        return "Apply(%s, %d)" % (repr(self.function), self.n)
