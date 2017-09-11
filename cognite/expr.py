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

    def get_shape(self):
        return self.function.get_output_shape(args)

    def assert_shape(self, shape):
        self.function.assert_output_shape(self.args, shape)

    def __repr__(self):
        return "Apply(%s, %s)" % (repr(self.function), repr(self.args))

class Constant(Expr):
    def __init__(self, value):
        self.value = value

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

    def __getitem__(self, attr):
        assert not self.shape
        descendent = self.descendents.get(attr)
        if descendent:
            return descendent

        name = '%s[%s]' % (self.name, repr(attr))
        descendent = Index(self, attr, name)
        self.descendents[attr] = descendent
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

class Index(Expr):
    def __init__(self, value, attr, name, shape=None, descendents=None):
        self.value = value
        self.attr = attr
        self.name = name
        self.shape = shape
        self.descendents = descendents or {}

    @property
    def children(self):
        return [self.value]

    def __getitem__(self, attr):
        assert not self.shape
        descendent = self.descendents.get(attr)
        if descendent:
            return descendent

        name = '%s[%s]' % (self.name, repr(attr))
        descendent = Index(self, attr, name)
        self.descendents[attr] = descendent
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
                raise ShapeError('Shape mismatch, %s cannot be both %s and %s' % (self.name, self.shape, shape))
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
