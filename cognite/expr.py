import collections

class ShapeError(Exception):
    pass

class Function:
    def forward(self, args):
        raise NotImplementedError()

    def assert_output_shape(self, args, shape):
        output_shape = self.get_output_shape(args)
        if output_shape != shape:
            raise expr.ShapeError('Expected %s, but got %s' % (shape, output_shape))

    def get_output_shape(self, args):
        raise NotImplementedError()

class Expr:
    @property
    def children(self):
        raise NotImplementedError()

    def get_shape(self):
        raise NotImplementedError()

    def assert_shape(self):
        raise NotImplementedError()

class Apply(Expr):
    def __init__(self, function, args):
        self.function = function
        self.args = args

    @property
    def children(self):
        return self.args

    def replace(self, replacements):
        new_args = [arg.replace(replacements) for arg in self.args]
        if new_args == self.args:
            return self
        elif all([isinstance(arg, Constant) for arg in new_args]):
            return Constant(self.function.forward([arg.value for arg in new_args])[0])
        else:
            return Apply(self.function, new_args)

    def get_shape(self):
        return self.function.get_output_shape(args)

    def assert_shape(self, shape):
        self.function.assert_output_shape(self.args, shape)

    def __repr__(self):
        return "Apply(%s, %s)" % (repr(self.function), repr(self.args))

class Constant(Expr):
    def __init__(self, value):
        self.value = value

    def replace(self, replacements):
        return self

    @property
    def children(self):
        return []

    def get_shape(self):
        return self.value.shape

    def assert_shape(self, shape):
        assert self.value.shape == shape

    def __repr__(self):
        return repr(self.value)

class Variable(Expr):
    def __init__(self, name):
        self.name = name
        self.descendents = {}
        self.shape = None

    @property
    def children(self):
        return []

    def replace(self, replacements):
        return replacements.get(self, self)

    def __getitem__(self, name):
        assert not self.shape
        descendent = self.descendents.get(name)
        if descendent:
            return descendent
        descendent = Variable('%s[%s]' % (self.name, repr(name)))
        self.descendents[name] = descendent
        return descendent

    def get_shape(self):
        if self.shape:
            return self.shape
        else:
            raise ShapeError('Shape not fixed for %s' % self.name)

    def assert_shape(self, shape):
        assert not self.descendents
        if self.shape:
            if self.shape != shape:
                raise ShapeError(
                    'Shape mismatch, %s cannot be both %s and %s' % (self.name, self.shape, shape)
                )
        else:
            self.shape = shape

    def __repr__(self):
        return self.name

def topological_sort(root):
    exprs = []
    stack = [root]
    while stack:
        x = stack.pop()
        if not x in exprs:
            stack.extend(x.children)
            exprs.append(x)
    exprs.reverse()
    return exprs

def count_references(exprs):
    refs = collections.Counter()
    for expr in exprs:
        for c in expr.children:
            refs[c] += 1
    return refs
